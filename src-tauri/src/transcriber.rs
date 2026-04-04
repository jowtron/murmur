use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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
        }
    }

    pub fn url(&self) -> String {
        let base = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
        format!("{}/{}", base, self.filename())
    }
}

pub fn models_dir() -> PathBuf {
    let dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("whisper-transcriber")
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

/// Convert audio file to 16kHz mono f32 PCM using ffmpeg
pub fn audio_to_pcm(path: &Path) -> Result<Vec<f32>, String> {
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-i",
            path.to_str().ok_or("Invalid path")?,
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "pipe:1",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run ffmpeg: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("ffmpeg error: {}", stderr));
    }

    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(samples)
}

/// Get audio duration in seconds using ffprobe
pub fn get_duration(path: &Path) -> Result<f64, String> {
    let output = std::process::Command::new("ffprobe")
        .args([
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path.to_str().ok_or("Invalid path")?,
        ])
        .output()
        .map_err(|e| format!("ffprobe error: {}", e))?;

    String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<f64>()
        .map_err(|e| format!("Failed to parse duration: {}", e))
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
        let samples = audio_to_pcm(audio_path)?;
        let duration_secs = samples.len() as f64 / 16000.0;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.thread_count);
        params.set_language(Some("en"));
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_token_timestamps(true);

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
