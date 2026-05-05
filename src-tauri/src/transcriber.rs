use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WhisperModel {
    Tiny,
    Base,
    Small,
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

/// Decode audio file using symphonia, resample to 16kHz mono f32 PCM
pub fn audio_to_pcm(path: &Path) -> Result<Vec<f32>, String> {
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

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("Failed to create decoder: {}", e))?;

    let mut all_samples: Vec<f32> = Vec::new();

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
            all_samples.push(sum / num_channels as f32);
        }
    }

    // Resample to 16kHz using sinc interpolation (rubato)
    let target_rate = 16000.0;
    if (source_rate - target_rate).abs() < 1.0 {
        return Ok(all_samples);
    }

    use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction, Resampler};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        target_rate / source_rate,
        2.0,
        params,
        all_samples.len(),
        1,
    ).map_err(|e| format!("Failed to create resampler: {}", e))?;

    let result = resampler.process(&[&all_samples], None)
        .map_err(|e| format!("Resampling failed: {}", e))?;

    Ok(result.into_iter().next().unwrap_or_default())
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
        progress_cb: Option<Arc<Mutex<dyn FnMut(f32) + Send>>>,
    ) -> Result<TranscriptionResult, String> {
        self.transcribe_inner(audio_path, progress_cb, false)
    }

    pub fn transcribe_per_word(
        &self,
        audio_path: &Path,
        progress_cb: Option<Arc<Mutex<dyn FnMut(f32) + Send>>>,
    ) -> Result<TranscriptionResult, String> {
        self.transcribe_inner(audio_path, progress_cb, true)
    }

    fn transcribe_inner(
        &self,
        audio_path: &Path,
        progress_cb: Option<Arc<Mutex<dyn FnMut(f32) + Send>>>,
        per_word: bool,
    ) -> Result<TranscriptionResult, String> {
        let samples = audio_to_pcm(audio_path)?;
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
                    f(progress as f32 / 100.0);
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
