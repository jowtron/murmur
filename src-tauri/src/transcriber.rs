use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use rubato::Resampler;
use symphonia::core::probe::Hint;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    pub file: String,
    pub segments: Vec<Segment>,
    pub text: String,
    pub duration_secs: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionProgress {
    pub job_id: String,
    pub file: String,
    pub progress: f32,
    pub status: String,
    /// How fast the current stage is running relative to realtime -- audio
    /// seconds processed per wall-clock second. `None` until there is enough
    /// of a sample to mean anything.
    #[serde(default)]
    pub speed: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
    SmallQ5_1,
    SmallQ8_0,
    Medium,
    LargeV3,
    LargeV3Turbo,
    DistilSmallEn,
    DistilMediumEn,
    DistilLargeV3,
}

impl WhisperModel {
    pub fn filename(&self) -> &str {
        match self {
            WhisperModel::Tiny => "ggml-tiny.bin",
            WhisperModel::Base => "ggml-base.bin",
            WhisperModel::Small => "ggml-small.bin",
            WhisperModel::SmallQ5_1 => "ggml-small-q5_1.bin",
            WhisperModel::SmallQ8_0 => "ggml-small-q8_0.bin",
            WhisperModel::Medium => "ggml-medium.bin",
            WhisperModel::LargeV3 => "ggml-large-v3.bin",
            WhisperModel::LargeV3Turbo => "ggml-large-v3-turbo.bin",
            WhisperModel::DistilSmallEn => "ggml-distil-small.en.bin",
            WhisperModel::DistilMediumEn => "ggml-distil-medium.en.bin",
            WhisperModel::DistilLargeV3 => "ggml-distil-large-v3.bin",
        }
    }

    pub fn display_name(&self) -> &str {
        match self {
            WhisperModel::Tiny => "Tiny (~75 MB)",
            WhisperModel::Base => "Base (~142 MB)",
            WhisperModel::Small => "Small (~466 MB)",
            WhisperModel::SmallQ5_1 => "Small Q5_1 (~181 MB, quantized)",
            WhisperModel::SmallQ8_0 => "Small Q8_0 (~252 MB, quantized)",
            WhisperModel::Medium => "Medium (~1.5 GB)",
            WhisperModel::LargeV3 => "Large V3 (~3 GB)",
            WhisperModel::LargeV3Turbo => "Large V3 Turbo (~1.6 GB)",
            WhisperModel::DistilSmallEn => "Distil Small.en (~336 MB, English-only)",
            WhisperModel::DistilMediumEn => "Distil Medium.en (~794 MB, English-only)",
            WhisperModel::DistilLargeV3 => "Distil Large V3 (~1.5 GB, English-only)",
        }
    }

    pub fn url(&self) -> String {
        match self {
            // Whisper baseline models — hosted on the canonical whisper.cpp repo
            WhisperModel::Tiny
            | WhisperModel::Base
            | WhisperModel::Small
            | WhisperModel::SmallQ5_1
            | WhisperModel::SmallQ8_0
            | WhisperModel::Medium
            | WhisperModel::LargeV3
            | WhisperModel::LargeV3Turbo => format!(
                "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}",
                self.filename()
            ),
            // Distil-Whisper variants — hosted on distil-whisper org
            WhisperModel::DistilSmallEn => {
                "https://huggingface.co/distil-whisper/distil-small.en/resolve/main/ggml-distil-small.en.bin".to_string()
            }
            WhisperModel::DistilMediumEn => {
                // distil-whisper repo uses an unconventional upstream filename
                "https://huggingface.co/distil-whisper/distil-medium.en/resolve/main/ggml-medium-32-2.en.bin".to_string()
            }
            WhisperModel::DistilLargeV3 => {
                "https://huggingface.co/distil-whisper/distil-large-v3-ggml/resolve/main/ggml-distil-large-v3.bin".to_string()
            }
        }
    }
}

pub fn models_dir() -> PathBuf {
    let dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur")
        .join("models");
    std::fs::create_dir_all(&dir).ok();
    dir
}

pub fn model_path(model: &WhisperModel) -> PathBuf {
    models_dir().join(model.filename())
}

pub fn is_model_downloaded(model: &WhisperModel) -> bool {
    model_path(model).exists()
}

/// Input frames handed to the resampler per call. Large enough that per-call
/// overhead is negligible, small enough that its buffers stay trivial next to
/// the decoded audio.
const RESAMPLE_CHUNK: usize = 16384;

/// Progress sink shared by the decode and transcribe phases: a completion
/// fraction for the *current* stage, plus the stage's name. Callers decide how
/// to weight the stages against each other, since each pipeline runs a
/// different set of them.
pub type StageProgress = Arc<Mutex<dyn FnMut(f32, &str) + Send>>;

/// Decode audio file using symphonia, resample to 16kHz mono f32 PCM
pub fn audio_to_pcm(path: &Path) -> Result<Vec<f32>, String> {
    audio_to_pcm_with_progress(path, None)
}

/// As `audio_to_pcm`, reporting decode and resample progress. Long audiobooks
/// spend minutes in here before the transcriber emits anything, so without this
/// the UI sits at 0% looking hung.
pub fn audio_to_pcm_with_progress(
    path: &Path,
    progress: Option<StageProgress>,
) -> Result<Vec<f32>, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open audio file: {}", e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("Failed to probe audio format: {}", e))?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("No audio track found")?;

    let codec_params = track.codec_params.clone();
    let track_id = track.id;
    let source_rate = codec_params.sample_rate.unwrap_or(44100) as f64;

    // Total frames lets us turn "samples decoded so far" into a percentage.
    // Absent it (some streamed containers) we just report the stage with no
    // fraction rather than inventing one.
    let total_frames = codec_params.n_frames.unwrap_or(0) as f64;

    let report = |frac: f32, stage: &str| {
        if let Some(cb) = &progress {
            if let Ok(mut f) = cb.lock() {
                f(frac, stage);
            }
        }
    };
    report(0.0, "decoding");

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    // Resample as we decode rather than buffering the whole file at its source
    // rate first. A 7.6-hour audiobook at 44.1 kHz mono f32 is ~4.8 GB held all
    // at once, and rubato's one-shot call over it reported no progress and
    // could not be interrupted. Streaming keeps only a small staging buffer
    // plus the 16 kHz output the caller actually needs.
    let target_rate = 16000.0;
    let needs_resample = (source_rate - target_rate).abs() >= 1.0;

    let mut resampler = if needs_resample {
        use rubato::{
            SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
        };
        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 256,
            window: WindowFunction::BlackmanHarris2,
        };
        Some(
            SincFixedIn::<f32>::new(target_rate / source_rate, 2.0, params, RESAMPLE_CHUNK, 1)
                .map_err(|e| format!("Failed to create resampler: {}", e))?,
        )
    } else {
        None
    };

    // Decoded mono frames waiting to fill the resampler's next input chunk.
    let mut staging: Vec<f32> = Vec::with_capacity(RESAMPLE_CHUNK * 2);
    let mut out: Vec<f32> = Vec::new();
    if total_frames > 0.0 {
        let ratio = if needs_resample { target_rate / source_rate } else { 1.0 };
        out.reserve((total_frames * ratio) as usize + RESAMPLE_CHUNK);
    }
    let mut decoded_frames: u64 = 0;
    // Emitting on every packet would be thousands of events a second; a packet
    // is ~20-40 ms of audio, so throttle to whole percentage points.
    let mut last_reported_pct = -1i32;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(format!("Error reading packet: {}", e)),
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(format!("Decode error: {}", e)),
        };

        let spec = *decoded.spec();
        let num_frames = decoded.frames();
        let mut sample_buf = SampleBuffer::<f32>::new(num_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        let samples = sample_buf.samples();

        // Mix down to mono
        let num_channels = spec.channels.count();
        for frame in 0..num_frames {
            let mut sum = 0.0f32;
            for ch in 0..num_channels {
                sum += samples[frame * num_channels + ch];
            }
            staging.push(sum / num_channels as f32);
        }
        decoded_frames += num_frames as u64;

        // Feed the resampler whole chunks; the remainder stays staged for the
        // next packet. Reusing one resampler across calls preserves its sinc
        // delay line, so the output is continuous across chunk boundaries.
        if let Some(resampler) = resampler.as_mut() {
            loop {
                let want = resampler.input_frames_next();
                if staging.len() < want {
                    break;
                }
                let mut chunk = resampler
                    .process(&[&staging[..want]], None)
                    .map_err(|e| format!("Resampling failed: {}", e))?;
                if let Some(channel) = chunk.pop() {
                    out.extend(channel);
                }
                staging.drain(..want);
            }
        } else {
            out.append(&mut staging);
        }

        if total_frames > 0.0 {
            let frac = (decoded_frames as f64 / total_frames).clamp(0.0, 1.0);
            let pct = (frac * 100.0) as i32;
            if pct > last_reported_pct {
                last_reported_pct = pct;
                report(frac as f32, "decoding");
            }
        }
    }

    // Whatever is left is shorter than a full chunk; `process_partial` pads it.
    if let Some(resampler) = resampler.as_mut() {
        if !staging.is_empty() {
            let mut chunk = resampler
                .process_partial(Some(&[&staging[..]]), None)
                .map_err(|e| format!("Resampling failed: {}", e))?;
            if let Some(channel) = chunk.pop() {
                out.extend(channel);
            }
        }
        // That last call emits a whole chunk's worth of output regardless of
        // how short the real remainder was, leaving up to a chunk of
        // padding-derived silence on the end. Cut back to the length the input
        // actually implies, or the file reads as a fraction of a second longer
        // than it is.
        let expected = (decoded_frames as f64 * (target_rate / source_rate)).round() as usize;
        if out.len() > expected {
            out.truncate(expected);
        }
    } else {
        out.append(&mut staging);
    }

    report(1.0, "decoding");
    Ok(out)
}

/// Get audio duration in seconds using symphonia
pub fn get_duration(path: &Path) -> Result<f64, String> {
    let file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open audio file: {}", e))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())
        .map_err(|e| format!("Failed to probe audio: {}", e))?;

    let format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != symphonia::core::codecs::CODEC_TYPE_NULL)
        .ok_or("No audio track found")?;

    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100) as f64;
    let n_frames = track.codec_params.n_frames.unwrap_or(0) as f64;

    if n_frames > 0.0 {
        Ok(n_frames / sample_rate)
    } else {
        // Fallback: use time_base and duration from the track
        if let (Some(tb), Some(dur)) = (track.codec_params.time_base, track.codec_params.n_frames) {
            Ok(tb.calc_time(dur).seconds as f64 + tb.calc_time(dur).frac)
        } else {
            Err("Could not determine duration".to_string())
        }
    }
}

pub struct Transcriber {
    ctx: WhisperContext,
    thread_count: i32,
}

impl Transcriber {
    pub fn new(model: &WhisperModel, thread_count: i32) -> Result<Self, String> {
        let path = model_path(model);
        if !path.exists() {
            return Err(format!("Model not found: {:?}", path));
        }

        let params = WhisperContextParameters::default();

        let ctx = WhisperContext::new_with_params(
            path.to_str().ok_or("Invalid model path")?,
            params,
        )
        .map_err(|e| format!("Failed to load model: {}", e))?;

        Ok(Self { ctx, thread_count })
    }

    pub fn transcribe(
        &self,
        audio_path: &Path,
        progress_cb: Option<StageProgress>,
    ) -> Result<TranscriptionResult, String> {
        self.transcribe_inner(audio_path, progress_cb, false)
    }

    pub fn transcribe_per_word(
        &self,
        audio_path: &Path,
        progress_cb: Option<StageProgress>,
    ) -> Result<TranscriptionResult, String> {
        self.transcribe_inner(audio_path, progress_cb, true)
    }

    fn transcribe_inner(
        &self,
        audio_path: &Path,
        progress_cb: Option<StageProgress>,
        per_word: bool,
    ) -> Result<TranscriptionResult, String> {
        let samples = audio_to_pcm_with_progress(audio_path, progress_cb.clone())?;
        let duration_secs = samples.len() as f64 / 16000.0;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.thread_count);
        params.set_language(Some("en"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);
        params.set_split_on_word(true);      // split at word boundaries, not mid-word
        params.set_max_len(if per_word { 1 } else { 20 });

        if let Some(cb) = progress_cb {
            params.set_progress_callback_safe(move |progress| {
                if let Ok(mut f) = cb.lock() {
                    f(progress as f32 / 100.0, "transcribing");
                }
            });
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;

        state
            .full(params, &samples)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        let num_segments = state.full_n_segments().map_err(|e| format!("{}", e))?;
        let mut segments = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_segments {
            let start = state.full_get_segment_t0(i).map_err(|e| format!("{}", e))? as f64 / 100.0;
            let end = state.full_get_segment_t1(i).map_err(|e| format!("{}", e))? as f64 / 100.0;
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| format!("{}", e))?;

            full_text.push_str(&text);
            segments.push(Segment {
                start,
                end,
                text: text.trim().to_string(),
            });
        }

        Ok(TranscriptionResult {
            file: audio_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            segments,
            text: full_text.trim().to_string(),
            duration_secs,
        })
    }
}

/// Format transcription as SRT subtitle format
/// Plain-text transcript with chapter headings interleaved at their start
/// times. Used when the source file states its own chapters (an .m4b
/// audiobook), so the transcript reads as a chaptered document instead of one
/// undifferentiated wall of text.
///
/// A heading lands before the first segment that starts at or after the
/// chapter's start time, so a chapter whose boundary falls mid-segment opens on
/// the next whole segment rather than splitting a sentence.
pub fn to_chaptered_text(result: &TranscriptionResult, chapters: &[(String, f64)]) -> String {
    // Chapters starting past the end of the audio are stale metadata -- a
    // trimmed file whose chapter list was copied wholesale, say -- and would
    // otherwise pile up as headings with no text under them.
    let chapters: Vec<&(String, f64)> = chapters
        .iter()
        .filter(|(_, start)| *start <= result.duration_secs)
        .collect();
    if chapters.is_empty() {
        return result.text.clone();
    }

    let mut out = String::new();
    let mut next = 0usize;

    let push_heading = |out: &mut String, title: &str, secs: f64| {
        let total = secs.max(0.0) as u64;
        let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&format!("=== {} [{:02}:{:02}:{:02}] ===\n\n", title, h, m, s));
    };

    for segment in &result.segments {
        // Several chapters can precede one segment (short front matter, or a
        // chapter that contains no speech at all); emit each in turn.
        while next < chapters.len() && segment.start >= chapters[next].1 {
            push_heading(&mut out, &chapters[next].0, chapters[next].1);
            next += 1;
        }
        if !out.is_empty() && !out.ends_with('\n') && !out.ends_with(' ') {
            out.push(' ');
        }
        out.push_str(&segment.text);
    }

    // Any chapters starting past the last segment still belong in the file.
    while next < chapters.len() {
        push_heading(&mut out, &chapters[next].0, chapters[next].1);
        next += 1;
    }

    out.trim_end().to_string()
}

pub fn to_srt(result: &TranscriptionResult) -> String {
    let mut out = String::new();
    for (i, seg) in result.segments.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            format_timestamp_srt(seg.start),
            format_timestamp_srt(seg.end)
        ));
        out.push_str(&format!("{}\n\n", seg.text));
    }
    out
}

/// Format transcription as VTT subtitle format
pub fn to_vtt(result: &TranscriptionResult) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for seg in &result.segments {
        out.push_str(&format!(
            "{} --> {}\n",
            format_timestamp_vtt(seg.start),
            format_timestamp_vtt(seg.end)
        ));
        out.push_str(&format!("{}\n\n", seg.text));
    }
    out
}

fn format_timestamp_srt(secs: f64) -> String {
    let h = (secs / 3600.0) as u32;
    let m = ((secs % 3600.0) / 60.0) as u32;
    let s = (secs % 60.0) as u32;
    let ms = ((secs % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, ms)
}

fn format_timestamp_vtt(secs: f64) -> String {
    let h = (secs / 3600.0) as u32;
    let m = ((secs % 3600.0) / 60.0) as u32;
    let s = (secs % 60.0) as u32;
    let ms = ((secs % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02}.{:03}", h, m, s, ms)
}

#[cfg(test)]
mod chaptered_text_tests {
    use super::{to_chaptered_text, Segment, TranscriptionResult};

    fn result(segments: &[(f64, &str)]) -> TranscriptionResult {
        let segments: Vec<Segment> = segments
            .iter()
            .map(|(start, text)| Segment { start: *start, end: start + 5.0, text: text.to_string() })
            .collect();
        let text = segments.iter().map(|s| s.text.clone()).collect::<Vec<_>>().join(" ");
        TranscriptionResult { file: "book.m4b".into(), segments, text, duration_secs: 60.0 }
    }

    #[test]
    fn returns_plain_text_when_there_are_no_chapters() {
        let r = result(&[(0.0, "one"), (5.0, "two")]);
        assert_eq!(to_chaptered_text(&r, &[]), "one two");
    }

    #[test]
    fn opens_each_chapter_on_the_next_whole_segment() {
        let r = result(&[(0.0, "intro line"), (10.0, "first line"), (20.0, "second line")]);
        let chapters = [("Opening".to_string(), 0.0), ("Chapter One".to_string(), 8.0)];
        let out = to_chaptered_text(&r, &chapters);
        assert_eq!(
            out,
            "=== Opening [00:00:00] ===\n\nintro line\n\n=== Chapter One [00:00:08] ===\n\nfirst line second line"
        );
    }

    #[test]
    fn emits_consecutive_chapters_that_contain_no_speech() {
        let r = result(&[(30.0, "late start")]);
        let chapters = [
            ("Credits".to_string(), 0.0),
            ("Dedication".to_string(), 10.0),
            ("Chapter One".to_string(), 20.0),
        ];
        let out = to_chaptered_text(&r, &chapters);
        assert!(out.contains("=== Credits [00:00:00] ==="));
        assert!(out.contains("=== Dedication [00:00:10] ==="));
        assert!(out.ends_with("=== Chapter One [00:00:20] ===\n\nlate start"));
    }

    #[test]
    fn keeps_a_trailing_chapter_inside_the_audio_but_past_the_last_segment() {
        // "End Credits" with no transcribed speech under it still belongs.
        let r = result(&[(0.0, "body")]);
        let chapters = [("Body".to_string(), 0.0), ("End Credits".to_string(), 55.0)];
        let out = to_chaptered_text(&r, &chapters);
        assert!(out.ends_with("=== End Credits [00:00:55] ==="));
    }

    #[test]
    fn drops_chapters_starting_past_the_end_of_the_audio() {
        // A trimmed file that kept the original chapter list would otherwise
        // end in a run of empty headings.
        let r = result(&[(0.0, "body")]);
        let chapters = [("Body".to_string(), 0.0), ("Stale".to_string(), 3600.0)];
        let out = to_chaptered_text(&r, &chapters);
        assert_eq!(out, "=== Body [00:00:00] ===\n\nbody");
    }
}
