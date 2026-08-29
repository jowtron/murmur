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
    /// Set when `repetition_loop_ranges` had to drop runaway repeats, so the
    /// caller can tell the user part of the audio went untranscribed. Optional
    /// and skipped when empty to keep the .json output shape unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_warning: Option<String>,
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
        let duration_secs = samples.len() as f64 / SAMPLE_RATE as f64;

        let bounds = chunk_boundaries(&samples);
        let chunk_count = bounds.len() - 1;

        let mut segments: Vec<Segment> = Vec::new();
        let mut raw_texts: Vec<String> = Vec::new();
        let mut unrecovered: Vec<(f64, f64, usize)> = Vec::new();

        for c in 0..chunk_count {
            let (from, to) = (bounds[c], bounds[c + 1]);

            // Each attempt is a perturbation, not just a retry. The collapse is
            // chaotically input-sensitive — a sub-sample resampling difference
            // was enough to flip the same audio between looping and clean — so
            // decoding the same samples differently is usually enough to escape
            // it. Ordered cheapest first; we stop at the first clean result.
            let attempts: [(SamplingStrategy, usize); 3] = [
                (SamplingStrategy::Greedy { best_of: 5 }, 0),
                (SamplingStrategy::BeamSearch { beam_size: 5, patience: -1.0 }, 0),
                (SamplingStrategy::Greedy { best_of: 5 }, SAMPLE_RATE / 8),
            ];

            let mut best: Option<(Vec<Segment>, Vec<String>, usize, f64)> = None;

            for (strategy, skip) in attempts {
                let begin = (from + skip).min(to);
                if begin >= to {
                    continue;
                }
                let offset = begin as f64 / SAMPLE_RATE as f64;
                let label = format!("{:?}", strategy);
                let prog = progress_cb.clone().map(|cb| (cb, c, chunk_count));
                let (segs, raw) = self.run_window(&samples[begin..to], per_word, strategy, prog)?;

                let lost: usize = repetition_loop_ranges(&segs).iter().map(|r| r.len()).sum();
                if lost > 0 {
                    // Worth seeing in logs: a chunk that needs retrying is the
                    // early warning that a file is prone to this.
                    eprintln!(
                        "murmur: chunk {}/{} looped ({} segments) with {:?}; perturbing",
                        c + 1,
                        chunk_count,
                        lost,
                        label
                    );
                }
                let better = best.as_ref().map_or(true, |(_, _, b, _)| lost < *b);
                if better {
                    best = Some((segs, raw, lost, offset));
                }
                if lost == 0 {
                    break;
                }
            }

            let Some((mut segs, mut raw, lost, offset)) = best else {
                continue;
            };

            // Whatever survived is still the best of the attempts; collapse any
            // residual loop so the transcript does not silently repeat itself.
            let dropped = repetition_loop_ranges(&segs);
            if !dropped.is_empty() {
                let keep = keep_mask(segs.len(), &dropped);
                segs = segs.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| x).collect();
                raw = raw.into_iter().zip(&keep).filter(|(_, k)| **k).map(|(x, _)| x).collect();
                unrecovered.push((offset, to as f64 / SAMPLE_RATE as f64, lost));
            }

            for mut seg in segs {
                seg.start += offset;
                seg.end += offset;
                segments.push(seg);
            }
            raw_texts.extend(raw);
        }

        Ok(TranscriptionResult {
            file: audio_path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            segments,
            text: raw_texts.concat().trim().to_string(),
            duration_secs,
            repetition_warning: describe_unrecovered(&unrecovered),
        })
    }

    /// One bounded `whisper_full` run over `pcm`, with a fresh state so no
    /// decoder state can carry across chunk boundaries.
    fn run_window(
        &self,
        pcm: &[f32],
        per_word: bool,
        strategy: SamplingStrategy,
        progress: Option<(StageProgress, usize, usize)>,
    ) -> Result<(Vec<Segment>, Vec<String>), String> {
        let mut params = FullParams::new(strategy);
        params.set_n_threads(self.thread_count);
        params.set_language(Some("en"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);
        params.set_split_on_word(true);
        params.set_max_len(if per_word { 1 } else { 20 });

        if let Some((cb, index, total)) = progress {
            params.set_progress_callback_safe(move |p| {
                if let Ok(mut f) = cb.lock() {
                    // Map this chunk's 0-100 onto its slice of the whole job.
                    // A retry restarts the chunk, so this can step backwards;
                    // the Pacer clamps the bar monotonic, so it never shows.
                    let overall = (index as f32 + (p as f32 / 100.0)) / total as f32;
                    f(overall, "transcribing");
                }
            });
        }

        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| format!("Failed to create state: {}", e))?;

        state
            .full(params, pcm)
            .map_err(|e| format!("Transcription failed: {}", e))?;

        let n = state.full_n_segments().map_err(|e| format!("{}", e))?;
        let mut segments = Vec::new();
        let mut raw = Vec::new();
        for i in 0..n {
            let start = state.full_get_segment_t0(i).map_err(|e| format!("{}", e))? as f64 / 100.0;
            let end = state.full_get_segment_t1(i).map_err(|e| format!("{}", e))? as f64 / 100.0;
            let text = state.full_get_segment_text(i).map_err(|e| format!("{}", e))?;
            segments.push(Segment { start, end, text: text.trim().to_string() });
            raw.push(text);
        }
        Ok((segments, raw))
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

pub const SAMPLE_RATE: usize = 16000;

/// Cap on how long a single `whisper_full` run may be.
///
/// The decoder can collapse into a repetition loop and then never recover for
/// the rest of the run — in every observed case it repeated one phrase to
/// end-of-file rather than resuming. Measured on one 98-minute recording:
/// greedy lost the last 12 minutes, beam search lost the last 45. Bounding the
/// run means a collapse can only ever cost part of one chunk. 5 minutes is
/// chosen because a loop was seen starting 11 minutes into a 17-minute run.
const CHUNK_SECS: f64 = 300.0;
/// How far either side of a boundary to hunt for a quiet place to cut.
const BOUNDARY_SEARCH_SECS: f64 = 20.0;
/// Width of the window scored while hunting for that quiet place.
const BOUNDARY_WINDOW_SECS: f64 = 0.4;
/// A trailing stub shorter than this is merged into the previous chunk.
const MIN_TAIL_SECS: f64 = 30.0;

/// Split points for `samples`, always starting at 0 and ending at `len()`.
///
/// Cuts land at the quietest point near each target boundary rather than on an
/// exact multiple, so a chunk edge is unlikely to fall mid-word.
fn chunk_boundaries(samples: &[f32]) -> Vec<usize> {
    let chunk = (CHUNK_SECS * SAMPLE_RATE as f64) as usize;
    if samples.len() <= chunk {
        return vec![0, samples.len()];
    }

    let search = (BOUNDARY_SEARCH_SECS * SAMPLE_RATE as f64) as usize;
    let win = (BOUNDARY_WINDOW_SECS * SAMPLE_RATE as f64) as usize;
    let min_tail = (MIN_TAIL_SECS * SAMPLE_RATE as f64) as usize;

    let mut out = vec![0usize];
    let mut target = chunk;

    while target + min_tail < samples.len() {
        let prev = *out.last().unwrap();
        let lo = target.saturating_sub(search).max(prev + win);
        let hi = (target + search).min(samples.len().saturating_sub(win));
        let mut cut = target;
        if lo < hi {
            let mut best = f64::MAX;
            let mut i = lo;
            while i + win <= hi {
                let e: f64 = samples[i..i + win].iter().map(|v| (*v as f64) * (*v as f64)).sum();
                if e < best {
                    best = e;
                    cut = i + win / 2;
                }
                i += win / 4;
            }
        }
        if cut <= prev {
            break;
        }
        out.push(cut);
        target = cut + chunk;
    }

    out.push(samples.len());
    out
}

/// Summarise chunks that were still looping after every retry.
fn describe_unrecovered(spans: &[(f64, f64, usize)]) -> Option<String> {
    if spans.is_empty() {
        return None;
    }
    let total: usize = spans.iter().map(|(_, _, n)| n).sum();
    let where_ = spans
        .iter()
        .map(|(a, b, _)| format!("{}-{}", format_timestamp_vtt(*a), format_timestamp_vtt(*b)))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "{} chunk(s) still looped after retries ({}); {} repeated segment(s) removed. That audio is not transcribed.",
        spans.len(),
        where_,
        total
    ))
}

/// Longest repeating block treated as a decoder loop, in segments.
const REPEAT_MAX_PERIOD: usize = 8;
/// Consecutive copies before we call it a loop rather than emphatic speech.
const REPEAT_MIN_COPIES: usize = 6;
/// Copies left in place, so the transcript still reads as what was said.
const REPEAT_KEEP_COPIES: usize = 2;

fn normalize_for_repeat(text: &str) -> String {
    text.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Find runs where a block of up to `REPEAT_MAX_PERIOD` segments repeats
/// consecutively at least `REPEAT_MIN_COPIES` times, and return the ranges
/// holding the surplus copies.
///
/// Whisper can collapse into a repetition loop and emit one phrase for the
/// rest of the file. `best_of: 5` in `transcribe_inner` is the real fix, but
/// no decoder setting removes the failure mode outright, and a transcript that
/// silently stops saying anything is worse than one that says it broke — so
/// this runs regardless. Matching is on a whole block, not a single segment,
/// because `set_max_len` splits one repeated phrase across several segments.
pub fn repetition_loop_ranges(segments: &[Segment]) -> Vec<std::ops::Range<usize>> {
    let norm: Vec<String> = segments.iter().map(|s| normalize_for_repeat(&s.text)).collect();
    let n = norm.len();
    let mut out = Vec::new();
    let mut i = 0;

    while i < n {
        // Prefer whichever period covers the most ground; on a tie the
        // smaller period wins, so a 3-segment phrase is not reported as a
        // 6-segment one repeating half as often.
        let mut best: Option<(usize, usize)> = None;
        for p in 1..=REPEAT_MAX_PERIOD.min(n - i) {
            if norm[i..i + p].iter().all(|t| t.is_empty()) {
                continue;
            }
            let mut copies = 1;
            while i + (copies + 1) * p <= n
                && norm[i..i + p] == norm[i + copies * p..i + (copies + 1) * p]
            {
                copies += 1;
            }
            if copies >= REPEAT_MIN_COPIES && best.map_or(true, |(bp, bc)| copies * p > bc * bp) {
                best = Some((p, copies));
            }
        }

        match best {
            Some((p, copies)) => {
                let end = i + copies * p;
                out.push(i + REPEAT_KEEP_COPIES * p..end);
                i = end;
            }
            None => i += 1,
        }
    }
    out
}

fn keep_mask(len: usize, dropped: &[std::ops::Range<usize>]) -> Vec<bool> {
    let mut keep = vec![true; len];
    for r in dropped {
        for k in r.clone() {
            keep[k] = false;
        }
    }
    keep
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
        TranscriptionResult { file: "book.m4b".into(), segments, text, duration_secs: 60.0, repetition_warning: None }
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

#[cfg(test)]
mod repetition_tests {
    use super::{repetition_loop_ranges, Segment, REPEAT_KEEP_COPIES};

    fn segs(texts: &[&str]) -> Vec<Segment> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| Segment {
                start: i as f64 * 2.0,
                end: i as f64 * 2.0 + 2.0,
                text: t.to_string(),
            })
            .collect()
    }

    fn dropped_count(texts: &[&str]) -> usize {
        repetition_loop_ranges(&segs(texts)).iter().map(|r| r.len()).sum()
    }

    #[test]
    fn leaves_ordinary_speech_alone() {
        let s = segs(&["hello there", "how are you", "fine thanks", "good good"]);
        assert!(repetition_loop_ranges(&s).is_empty());
    }

    #[test]
    fn tolerates_short_emphatic_repetition() {
        // "yeah yeah yeah yeah" is real speech, not a decoder loop.
        assert_eq!(dropped_count(&["yeah", "yeah", "yeah", "yeah"]), 0);
    }

    #[test]
    fn collapses_a_single_segment_loop() {
        let texts = vec!["intro"; 1]
            .into_iter()
            .chain(std::iter::repeat("i am not going to").take(50))
            .collect::<Vec<_>>();
        let ranges = repetition_loop_ranges(&segs(&texts));
        assert_eq!(ranges.len(), 1);
        // 50 copies in, REPEAT_KEEP_COPIES survive.
        assert_eq!(ranges[0].len(), 50 - REPEAT_KEEP_COPIES);
        assert_eq!(ranges[0].start, 1 + REPEAT_KEEP_COPIES);
    }

    #[test]
    fn collapses_a_multi_segment_phrase_loop() {
        // The real failure: max_len splits one phrase across 3 segments, and
        // the whole block repeats. Detection must work on the block, not the
        // individual segment.
        let mut texts: Vec<&str> = vec!["so there we go"];
        for _ in 0..40 {
            texts.extend_from_slice(&["i'm not going to", "be interested", "in you guys"]);
        }
        let ranges = repetition_loop_ranges(&segs(&texts));
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].len(), (40 - REPEAT_KEEP_COPIES) * 3);
    }

    #[test]
    fn ignores_case_and_whitespace_differences() {
        let texts = std::iter::repeat("The  Same   Thing")
            .take(4)
            .chain(std::iter::repeat("the same thing").take(4))
            .collect::<Vec<_>>();
        assert_eq!(dropped_count(&texts), 8 - REPEAT_KEEP_COPIES);
    }

    #[test]
    fn does_not_report_empty_segments_as_a_loop() {
        assert_eq!(dropped_count(&["", "", "", "", "", "", "", ""]), 0);
    }
}

#[cfg(test)]
mod long_file_check {
    use super::{Transcriber, WhisperModel};
    use std::path::Path;

    /// End-to-end check against a real recording. Ignored by default because
    /// it needs a multi-GB model and runs for minutes; the path comes from the
    /// environment so no personal file path lives in the repo.
    ///
    ///   MURMUR_LOOP_CHECK="/path/to/audio.m4a" \
    ///     cargo test --release long_file -- --ignored --nocapture
    #[test]
    #[ignore]
    fn long_file_transcribes_to_the_end_without_looping() {
        let path = std::env::var("MURMUR_LOOP_CHECK")
            .expect("set MURMUR_LOOP_CHECK to the audio file to check");
        let threads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(8);

        let t = Transcriber::new(&WhisperModel::LargeV3, threads).expect("load model");
        let r = t.transcribe(Path::new(&path), None).expect("transcribe");

        let last_end = r.segments.last().map(|s| s.end).unwrap_or(0.0);
        let hhmmss = |s: f64| {
            let s = s.max(0.0) as u64;
            format!("{:02}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
        };

        println!("audio duration : {} ({:.1}s)", hhmmss(r.duration_secs), r.duration_secs);
        println!("last segment   : {} ({:.1}s)", hhmmss(last_end), last_end);
        println!("coverage       : {:.1}%", last_end / r.duration_secs * 100.0);
        println!("segments       : {}", r.segments.len());
        println!("chars          : {}", r.text.len());
        match &r.repetition_warning {
            Some(w) => println!("GUARD FIRED    : {}", w),
            None => println!("GUARD          : clean, no loop detected"),
        }
        let tail = r.text.chars().rev().take(220).collect::<String>();
        println!("tail           : ...{}", tail.chars().rev().collect::<String>());

        // Assert on content, not on timestamps: the copies the guard leaves
        // behind carry timestamps near the end of the file, so a looped run
        // still reports full coverage. That mistake made an earlier version of
        // this test pass on a transcript that was missing 12 minutes.
        assert!(
            r.repetition_warning.is_none(),
            "decoder collapsed into a loop: {}",
            r.repetition_warning.clone().unwrap_or_default()
        );
        assert!(
            last_end >= r.duration_secs * 0.97,
            "transcript stops at {} of {}",
            hhmmss(last_end),
            hhmmss(r.duration_secs)
        );
    }
}

#[cfg(test)]
mod decode_dump {
    use super::audio_to_pcm;
    use std::io::Write;
    use std::path::Path;

    /// Write our own decode of a region to a wav, so it can be listened to or
    /// fed back through the model independently of ffmpeg's decode.
    ///
    ///   MURMUR_DUMP_SRC=in.m4a MURMUR_DUMP_OUT=out.wav MURMUR_DUMP_FROM=4900 \
    ///     cargo test --release decode_dump -- --ignored --nocapture
    #[test]
    #[ignore]
    fn dump_our_decoded_region() {
        let src = std::env::var("MURMUR_DUMP_SRC").expect("MURMUR_DUMP_SRC");
        let out = std::env::var("MURMUR_DUMP_OUT").expect("MURMUR_DUMP_OUT");
        let from: f64 = std::env::var("MURMUR_DUMP_FROM").expect("MURMUR_DUMP_FROM").parse().unwrap();

        let pcm = audio_to_pcm(Path::new(&src)).expect("decode");
        let start = ((from * 16000.0) as usize).min(pcm.len());
        let body = &pcm[start..];
        println!("dumping {} samples ({:.1}s) from {:.1}s", body.len(), body.len() as f64 / 16000.0, from);

        let (mut lo, mut hi, mut clipped) = (f32::MAX, f32::MIN, 0usize);
        for &v in body {
            lo = lo.min(v); hi = hi.max(v);
            if v.abs() > 1.0 { clipped += 1; }
        }
        println!("f32 range   : {:.4} .. {:.4}", lo, hi);
        println!("over +-1.0  : {} samples ({:.4}%)", clipped, clipped as f64 / body.len() as f64 * 100.0);

        let bytes: Vec<u8> = body
            .iter()
            .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect();
        let mut f = std::fs::File::create(&out).expect("create wav");
        let (sr, ch, bits) = (16000u32, 1u16, 16u16);
        f.write_all(b"RIFF").unwrap();
        f.write_all(&(36 + bytes.len() as u32).to_le_bytes()).unwrap();
        f.write_all(b"WAVEfmt ").unwrap();
        f.write_all(&16u32.to_le_bytes()).unwrap();
        f.write_all(&1u16.to_le_bytes()).unwrap();
        f.write_all(&ch.to_le_bytes()).unwrap();
        f.write_all(&sr.to_le_bytes()).unwrap();
        f.write_all(&(sr * ch as u32 * bits as u32 / 8).to_le_bytes()).unwrap();
        f.write_all(&(ch * bits / 8).to_le_bytes()).unwrap();
        f.write_all(&bits.to_le_bytes()).unwrap();
        f.write_all(b"data").unwrap();
        f.write_all(&(bytes.len() as u32).to_le_bytes()).unwrap();
        f.write_all(&bytes).unwrap();
        println!("wrote {}", out);
    }
}

#[cfg(test)]
mod chunk_boundary_tests {
    use super::{chunk_boundaries, CHUNK_SECS, MIN_TAIL_SECS, SAMPLE_RATE};

    fn secs(n: f64) -> usize {
        (n * SAMPLE_RATE as f64) as usize
    }

    /// Deterministic stand-in for speech: loud everywhere except the gaps.
    fn tone(len: usize, quiet: &[(usize, usize)]) -> Vec<f32> {
        let mut v: Vec<f32> = (0..len)
            .map(|i| ((i as f32) * 0.05).sin() * 0.5)
            .collect();
        for (a, b) in quiet {
            for s in v[*a..*b].iter_mut() {
                *s = 0.0;
            }
        }
        v
    }

    #[test]
    fn a_short_file_is_one_chunk() {
        let v = tone(secs(60.0), &[]);
        assert_eq!(chunk_boundaries(&v), vec![0, v.len()]);
    }

    #[test]
    fn boundaries_span_the_whole_file_and_increase() {
        let v = tone(secs(720.0), &[]);
        let b = chunk_boundaries(&v);
        assert_eq!(b[0], 0);
        assert_eq!(*b.last().unwrap(), v.len());
        assert!(b.len() >= 3, "720s should split into multiple chunks, got {:?}", b.len());
        assert!(b.windows(2).all(|w| w[0] < w[1]), "boundaries must increase: {:?}", b);
    }

    #[test]
    fn no_chunk_exceeds_the_cap() {
        let v = tone(secs(1300.0), &[]);
        let b = chunk_boundaries(&v);
        let longest = b.windows(2).map(|w| w[1] - w[0]).max().unwrap();
        // A cut may land up to BOUNDARY_SEARCH_SECS late, plus a merged tail.
        let ceiling = secs(CHUNK_SECS + 20.0 + MIN_TAIL_SECS);
        assert!(longest <= ceiling, "chunk of {} samples exceeds {}", longest, ceiling);
    }

    #[test]
    fn there_is_no_runt_chunk_at_the_end() {
        // 300s cap with a 310s file: the 10s tail must merge, not stand alone.
        let v = tone(secs(310.0), &[]);
        let b = chunk_boundaries(&v);
        let shortest = b.windows(2).map(|w| w[1] - w[0]).min().unwrap();
        assert!(shortest >= secs(MIN_TAIL_SECS), "runt chunk of {} samples", shortest);
    }

    #[test]
    fn the_cut_lands_in_a_quiet_gap_near_the_boundary() {
        // Silence from 292s to 294s; the cut for the 300s boundary should
        // prefer it over slicing through the tone.
        let gap = (secs(292.0), secs(294.0));
        let v = tone(secs(720.0), &[gap]);
        let b = chunk_boundaries(&v);
        let cut = b[1];
        assert!(
            cut >= gap.0 && cut <= gap.1,
            "cut at {:.1}s did not land in the 292-294s gap",
            cut as f64 / SAMPLE_RATE as f64
        );
    }
}
