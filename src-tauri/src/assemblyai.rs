use crate::curl_util::{auth_header_config, run_curl};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

const API_BASE: &str = "https://api.assemblyai.com/v2";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub start: u64,
    pub end: u64,
    pub text: String,
    pub confidence: Option<f64>,
    pub speaker: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    pub start: u64,
    pub end: u64,
    pub text: String,
    pub speaker: String,
    #[serde(default)]
    pub words: Vec<Word>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transcript {
    pub id: String,
    pub status: String,
    pub error: Option<String>,
    pub text: Option<String>,
    pub audio_duration: Option<f64>,
    #[serde(default)]
    pub utterances: Option<Vec<Utterance>>,
    #[serde(default)]
    pub words: Option<Vec<Word>>,
}

#[derive(Debug, Deserialize)]
struct UploadResponse {
    upload_url: String,
}

/// Upload a local file to AssemblyAI's /v2/upload endpoint.
/// Returns the temporary upload URL to pass to /v2/transcript.
/// Retries transient failures and aborts if the transfer stalls below
/// 10 KB/s for 60s (matches the model-download curl resilience). Killed
/// promptly if `cancel` fires — no point finishing a multi-GB upload for
/// a cancelled job.
pub async fn upload_file(
    path: &Path,
    api_key: &str,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let url = format!("{}/upload", API_BASE);
    let path_str = path
        .to_str()
        .ok_or_else(|| "Path is not valid UTF-8".to_string())?;
    let data_arg = format!("@{}", path_str);

    let stdout = run_curl(
        &[
            "--fail-with-body",
            "-X", "POST",
            "-H", "Content-Type: application/octet-stream",
            "--data-binary", &data_arg,
            "--connect-timeout", "30",
            "--retry", "5",
            "--retry-all-errors",
            "--speed-limit", "10000",
            "--speed-time", "60",
            &url,
        ],
        &auth_header_config(api_key),
        Some(cancel),
    )
    .await?;

    let parsed: UploadResponse = serde_json::from_slice(&stdout)
        .map_err(|e| format!("Failed to parse upload response: {} | body: {}", e, String::from_utf8_lossy(&stdout)))?;
    Ok(parsed.upload_url)
}

/// Submit a transcription request with speaker diarization enabled.
/// `speech_models` is the priority-routing array (e.g. ["universal-3-pro", "universal-2"]).
/// Returns the transcript ID for polling.
pub async fn submit(
    audio_url: &str,
    api_key: &str,
    language_code: Option<&str>,
    speech_models: &[String],
    speakers_expected: Option<u32>,
    cancel: &CancellationToken,
) -> Result<String, String> {
    let url = format!("{}/transcript", API_BASE);

    let mut body = serde_json::json!({
        "audio_url": audio_url,
        "speaker_labels": true,
        "punctuate": true,
        "format_text": true,
    });
    if let Some(lc) = language_code {
        body["language_code"] = serde_json::Value::String(lc.to_string());
    }
    // Hint the diarizer how many distinct speakers to expect. Known-count
    // recordings (e.g. a 2-person walkthrough) label far more cleanly with
    // this set. Clamped to AssemblyAI's accepted 1..=10 range; out-of-range
    // or 0 is treated as "auto" (omit the field).
    if let Some(n) = speakers_expected {
        if (1..=10).contains(&n) {
            body["speakers_expected"] = serde_json::Value::Number(n.into());
        }
    }
    if !speech_models.is_empty() {
        body["speech_models"] = serde_json::Value::Array(
            speech_models
                .iter()
                .map(|m| serde_json::Value::String(m.clone()))
                .collect(),
        );
    }
    let body_str = serde_json::to_string(&body).unwrap();

    let stdout = run_curl(
        &[
            "--fail-with-body",
            "-X", "POST",
            "-H", "Content-Type: application/json",
            "-d", &body_str,
            "--connect-timeout", "30",
            "--retry", "3",
            &url,
        ],
        &auth_header_config(api_key),
        Some(cancel),
    )
    .await?;

    let parsed: Transcript = serde_json::from_slice(&stdout)
        .map_err(|e| format!("Failed to parse submit response: {} | body: {}", e, String::from_utf8_lossy(&stdout)))?;
    Ok(parsed.id)
}

/// Fetch current transcript state. Status values: "queued", "processing", "completed", "error".
pub async fn poll(transcript_id: &str, api_key: &str) -> Result<Transcript, String> {
    let url = format!("{}/transcript/{}", API_BASE, transcript_id);

    let stdout = run_curl(
        &["--fail-with-body", "--connect-timeout", "30", "--max-time", "120", &url],
        &auth_header_config(api_key),
        None,
    )
    .await?;

    serde_json::from_slice(&stdout)
        .map_err(|e| format!("Failed to parse poll response: {} | body: {}", e, String::from_utf8_lossy(&stdout)))
}

pub fn poll_interval() -> Duration {
    Duration::from_secs(3)
}

fn fmt_srt_time(ms: u64) -> String {
    let total_secs = ms / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    let mmm = ms % 1000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, mmm)
}

/// Convert utterances to SRT with `Speaker A: ` prefix per cue.
pub fn utterances_to_srt(utts: &[Utterance]) -> String {
    let mut out = String::new();
    for (i, u) in utts.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", fmt_srt_time(u.start), fmt_srt_time(u.end)));
        out.push_str(&format!("Speaker {}: {}\n\n", u.speaker, u.text.trim()));
    }
    out
}

/// Plain text with one utterance per paragraph, prefixed by speaker label.
pub fn utterances_to_text(utts: &[Utterance]) -> String {
    let mut out = String::new();
    for u in utts {
        out.push_str(&format!("Speaker {}: {}\n\n", u.speaker, u.text.trim()));
    }
    out
}
