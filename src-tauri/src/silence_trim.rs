use std::io::Write;
use std::path::{Path, PathBuf};

/// One preserved speech chunk: the time in the trimmed audio, the time in the
/// original audio, and how long the chunk runs. Used to map AssemblyAI's
/// utterance timestamps (which are in the trimmed timeline) back to the
/// original file's timeline.
#[derive(Debug, Clone)]
pub struct SegmentMap {
    pub trimmed_start: f64,
    pub original_start: f64,
    pub duration: f64,
}

/// Find regions of speech in 16 kHz mono f32 PCM. Returns (start_secs, end_secs)
/// in the original audio timeline.
///
/// - `threshold_db`: frames with RMS energy below this dB level are considered
///   silent. -35 dB is a reasonable default for conversational speech.
/// - `min_silence_secs`: only silence runs at least this long are cut. Shorter
///   pauses (mid-sentence breaths) are preserved so we don't clip words.
/// - `padding_secs`: extend each retained speech region by this much on both
///   sides to absorb word onsets/offsets.
pub fn detect_speech_regions(
    samples_16k_mono: &[f32],
    threshold_db: f32,
    min_silence_secs: f32,
    padding_secs: f32,
) -> Vec<(f64, f64)> {
    let sr: usize = 16_000;
    let frame_size: usize = sr / 50; // 20 ms = 320 samples
    if samples_16k_mono.is_empty() {
        return Vec::new();
    }

    let total_secs = samples_16k_mono.len() as f64 / sr as f64;
    let n_frames = samples_16k_mono.len().div_ceil(frame_size);

    let mut is_silent = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * frame_size;
        let end = (start + frame_size).min(samples_16k_mono.len());
        let mut sum_sq = 0.0f64;
        for &s in &samples_16k_mono[start..end] {
            sum_sq += (s as f64) * (s as f64);
        }
        let rms = (sum_sq / (end - start) as f64).sqrt();
        let db = if rms > 1e-10 { 20.0 * rms.log10() } else { -120.0 };
        is_silent.push(db < threshold_db as f64);
    }

    let min_silence_frames = (min_silence_secs * 50.0).ceil() as usize;
    let mut keep_frame = vec![true; n_frames];
    let mut i = 0;
    while i < n_frames {
        if is_silent[i] {
            let mut j = i;
            while j < n_frames && is_silent[j] {
                j += 1;
            }
            if j - i >= min_silence_frames {
                for k in i..j {
                    keep_frame[k] = false;
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }

    let mut regions: Vec<(f64, f64)> = Vec::new();
    let mut current: Option<usize> = None;
    for k in 0..n_frames {
        if keep_frame[k] {
            if current.is_none() {
                current = Some(k);
            }
        } else if let Some(start) = current {
            let region_start = (start * frame_size) as f64 / sr as f64;
            let region_end = ((k * frame_size) as f64 / sr as f64).min(total_secs);
            regions.push((region_start, region_end));
            current = None;
        }
    }
    if let Some(start) = current {
        let region_start = (start * frame_size) as f64 / sr as f64;
        regions.push((region_start, total_secs));
    }

    let pad = padding_secs as f64;
    for r in regions.iter_mut() {
        r.0 = (r.0 - pad).max(0.0);
        r.1 = (r.1 + pad).min(total_secs);
    }

    let mut merged: Vec<(f64, f64)> = Vec::new();
    for r in regions {
        if let Some(last) = merged.last_mut() {
            if r.0 <= last.1 {
                last.1 = last.1.max(r.1);
                continue;
            }
        }
        merged.push(r);
    }

    merged
}

pub struct TrimResult {
    pub temp_audio: PathBuf,
    pub map: Vec<SegmentMap>,
    pub original_duration: f64,
    pub trimmed_duration: f64,
}

/// Decode the input file, detect speech, splice into a 16 kHz mono 16-bit FLAC
/// in the OS temp dir, and return the path along with the mapping needed to
/// translate AssemblyAI timestamps back to the original timeline.
pub fn trim_audio_to_temp_flac(
    audio_path: &Path,
    threshold_db: f32,
    min_silence_secs: f32,
    padding_secs: f32,
) -> Result<TrimResult, String> {
    let samples = crate::transcriber::audio_to_pcm(audio_path)?;
    let sr: usize = 16_000;
    let total_secs = samples.len() as f64 / sr as f64;

    let regions = detect_speech_regions(&samples, threshold_db, min_silence_secs, padding_secs);
    if regions.is_empty() {
        return Err("Silence trim removed all audio — try a lower dB threshold".to_string());
    }

    let mut out_samples: Vec<f32> = Vec::new();
    let mut map: Vec<SegmentMap> = Vec::new();
    let mut cursor: f64 = 0.0;
    for (rs, re) in &regions {
        let start_idx = (rs * sr as f64).floor() as usize;
        let end_idx = ((re * sr as f64).ceil() as usize).min(samples.len());
        if start_idx >= end_idx {
            continue;
        }
        let slice = &samples[start_idx..end_idx];
        let duration = (end_idx - start_idx) as f64 / sr as f64;
        map.push(SegmentMap {
            trimmed_start: cursor,
            original_start: *rs,
            duration,
        });
        out_samples.extend_from_slice(slice);
        cursor += duration;
    }
    let trimmed_duration = cursor;

    let temp_path = std::env::temp_dir()
        .join(format!("murmur_aai_{}.flac", uuid::Uuid::new_v4()));
    write_flac_16bit_mono_16k(&temp_path, &out_samples)?;

    Ok(TrimResult {
        temp_audio: temp_path,
        map,
        original_duration: total_secs,
        trimmed_duration,
    })
}

/// Decode the input file into a 16 kHz mono 16-bit FLAC in the OS temp dir,
/// with no trimming. Used for video containers so only the audio track is
/// uploaded to cloud engines, not the (much larger) video stream.
pub fn extract_audio_to_temp_flac(audio_path: &Path) -> Result<PathBuf, String> {
    let samples = crate::transcriber::audio_to_pcm(audio_path)?;
    let temp_path = std::env::temp_dir()
        .join(format!("murmur_extract_{}.flac", uuid::Uuid::new_v4()));
    write_flac_16bit_mono_16k(&temp_path, &samples)?;
    Ok(temp_path)
}

/// Translate a millisecond timestamp from the trimmed audio's timeline back to
/// the original audio's timeline using the segment map.
pub fn map_trimmed_ms_to_original_ms(trimmed_ms: u64, map: &[SegmentMap]) -> u64 {
    if map.is_empty() {
        return trimmed_ms;
    }
    let t = trimmed_ms as f64 / 1000.0;
    for seg in map {
        let seg_end = seg.trimmed_start + seg.duration;
        if t >= seg.trimmed_start && t < seg_end {
            let offset = t - seg.trimmed_start;
            return ((seg.original_start + offset) * 1000.0).round() as u64;
        }
    }
    // Past the end of the last segment — snap to its end.
    if let Some(last) = map.last() {
        return ((last.original_start + last.duration) * 1000.0).round() as u64;
    }
    trimmed_ms
}

/// Encode 16 kHz mono f32 samples as a 16-bit FLAC file.
///
/// FLAC rather than WAV because this is what gets uploaded to the cloud
/// engines: it is lossless (the services decode exactly the samples a WAV
/// would have carried) at roughly half the bytes, which on a slow uplink is
/// half the wait. Both AssemblyAI and Deepgram accept FLAC.
fn write_flac_16bit_mono_16k(path: &Path, samples: &[f32]) -> Result<(), String> {
    use flacenc::component::BitRepr;
    use flacenc::error::Verify;

    let config = flacenc::config::Encoder::default()
        .into_verified()
        .map_err(|e| format!("FLAC config error: {:?}", e))?;
    let source = F32Source { samples, pos: 0, scratch: Vec::new() };
    let mut stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
        .map_err(|e| format!("FLAC encode failed: {:?}", e))?;
    // flacenc counts the short final block in STREAMINFO's minimum block
    // size. The spec excludes it, and with min != max decoders read the
    // stream as variable-blocksize: libFLAC warns about frame numbering and
    // symphonia refuses to open the file at all. Restate it as fixed.
    let max_block = stream.stream_info().max_block_size();
    stream
        .stream_info_mut()
        .set_block_sizes(max_block, max_block)
        .map_err(|e| format!("FLAC header error: {:?}", e))?;
    let mut sink = flacenc::bitsink::ByteSink::new();
    stream
        .write(&mut sink)
        .map_err(|e| format!("FLAC write failed: {:?}", e))?;
    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("Failed to create temp flac: {}", e))?;
    file.write_all(sink.as_slice()).map_err(|e| e.to_string())?;
    Ok(())
}

/// Feeds f32 samples to the FLAC encoder a block at a time, converting to
/// 16-bit as it goes, so a multi-hour file isn't copied whole into i32 first.
struct F32Source<'a> {
    samples: &'a [f32],
    pos: usize,
    scratch: Vec<i32>,
}

impl flacenc::source::Source for F32Source<'_> {
    fn channels(&self) -> usize {
        1
    }
    fn bits_per_sample(&self) -> usize {
        16
    }
    fn sample_rate(&self) -> usize {
        16_000
    }
    fn read_samples<F: flacenc::source::Fill>(
        &mut self,
        block_size: usize,
        dest: &mut F,
    ) -> Result<usize, flacenc::error::SourceError> {
        let end = (self.pos + block_size).min(self.samples.len());
        self.scratch.clear();
        self.scratch.extend(
            self.samples[self.pos..end]
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32_767.0) as i32),
        );
        dest.fill_interleaved(&self.scratch)?;
        let n = end - self.pos;
        self.pos = end;
        Ok(n)
    }
    fn len_hint(&self) -> Option<usize> {
        Some(self.samples.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a real file, encode it as the upload FLAC, decode that back
    /// and require the 16-bit samples to match exactly. Path comes from the
    /// environment so no personal file is committed:
    ///   MURMUR_FLAC_CHECK=/path/audio.m4a cargo test --release flac_roundtrip -- --ignored --nocapture
    #[test]
    #[ignore]
    fn flac_roundtrip() {
        let src = std::env::var("MURMUR_FLAC_CHECK").expect("set MURMUR_FLAC_CHECK");
        let samples = crate::transcriber::audio_to_pcm(Path::new(&src)).unwrap();
        let out = std::env::temp_dir().join("murmur_flac_roundtrip.flac");
        let t = std::time::Instant::now();
        write_flac_16bit_mono_16k(&out, &samples).unwrap();
        let enc = t.elapsed();
        let flac_bytes = std::fs::metadata(&out).unwrap().len();
        let wav_bytes = 44 + samples.len() as u64 * 2;

        let file = std::fs::File::open(&out).unwrap();
        let mss = symphonia::core::io::MediaSourceStream::new(Box::new(file), Default::default());
        let probed = symphonia::default::get_probe()
            .format(&Default::default(), mss, &Default::default(), &Default::default())
            .unwrap();
        let mut format = probed.format;
        let track = format.default_track().unwrap().clone();
        assert_eq!(track.codec_params.sample_rate, Some(16_000));
        let mut dec = symphonia::default::get_codecs()
            .make(&track.codec_params, &Default::default())
            .unwrap();
        let mut decoded: Vec<i16> = Vec::with_capacity(samples.len());
        while let Ok(pkt) = format.next_packet() {
            let buf = dec.decode(&pkt).unwrap();
            let mut sb = symphonia::core::audio::SampleBuffer::<i16>::new(
                buf.capacity() as u64,
                *buf.spec(),
            );
            sb.copy_interleaved_ref(buf);
            decoded.extend_from_slice(sb.samples());
        }
        let expected: Vec<i16> = samples
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * 32_767.0) as i16)
            .collect();
        println!(
            "{:.0} s audio: encode {:.2?}, FLAC {:.1} MB vs WAV {:.1} MB ({:.0}%)",
            samples.len() as f64 / 16_000.0,
            enc,
            flac_bytes as f64 / 1e6,
            wav_bytes as f64 / 1e6,
            100.0 * flac_bytes as f64 / wav_bytes as f64
        );
        assert_eq!(decoded.len(), expected.len());
        assert!(decoded == expected, "decoded FLAC differs from source samples");
        let _ = std::fs::remove_file(&out);
    }
}
