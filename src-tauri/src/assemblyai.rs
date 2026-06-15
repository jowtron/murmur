use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use tokio::process::Command;

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

fn run_curl(args: &[&str]) -> Result<Vec<u8>, String> {
    let output = std::process::Command::new("curl")
        .args(args)
        .output()
        .map_err(|e| format!("curl failed to start: {}", e))?;
    if !output.status.success() {
        return Err(format!(
            "curl exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(output.stdout)
}

/// Upload a local file to AssemblyAI's /v2/upload endpoint.
/// Returns the temporary upload URL to pass to /v2/transcript.
pub fn upload_file(path: &Path, api_key: &str) -> Result<String, String> {
    let auth = format!("Authorization: {}", api_key);
    let content_type = "Content-Type: application/octet-stream";
    let url = format!("{}/upload", API_BASE);
    let path_str = path
        .to_str()
        .ok_or_else(|| "Path is not valid UTF-8".to_string())?;
    let data_arg = format!("@{}", path_str);

    let stdout = run_curl(&[
        "-sS",
        "--fail-with-body",
        "-X", "POST",
        "-H", &auth,
        "-H", content_type,
        "--data-binary", &data_arg,
        &url,
    ])?;

    let parsed: UploadResponse = serde_json::from_slice(&stdout)
        .map_err(|e| format!("Failed to parse upload response: {} | body: {}", e, String::from_utf8_lossy(&stdout)))?;
    Ok(parsed.upload_url)
}

/// Submit a transcription request with speaker diarization enabled.
/// `speech_models` is the priority-routing array (e.g. ["universal-3-pro", "universal-2"]).
/// Returns the transcript ID for polling.
pub fn submit(
    audio_url: &str,
    api_key: &str,
    language_code: Option<&str>,
    speech_models: &[String],
    speakers_expected: Option<u32>,
) -> Result<String, String> {
    let auth = format!("Authorization: {}", api_key);
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

    let stdout = run_curl(&[
        "-sS",
        "--fail-with-body",
        "-X", "POST",
        "-H", &auth,
        "-H", "Content-Type: application/json",
        "-d", &body_str,
        &url,
    ])?;

    let parsed: Transcript = serde_json::from_slice(&stdout)
        .map_err(|e| format!("Failed to parse submit response: {} | body: {}", e, String::from_utf8_lossy(&stdout)))?;
    Ok(parsed.id)
}

/// Fetch current transcript state. Status values: "queued", "processing", "completed", "error".
pub async fn poll(transcript_id: &str, api_key: &str) -> Result<Transcript, String> {
    let auth = format!("Authorization: {}", api_key);
    let url = format!("{}/transcript/{}", API_BASE, transcript_id);

    let output = Command::new("curl")
        .args(["-sS", "--fail-with-body", "-H", &auth, &url])
        .output()
        .await
        .map_err(|e| format!("curl failed to start: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "curl exited {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse poll response: {} | body: {}", e, String::from_utf8_lossy(&output.stdout)))
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

/// Hybrid "Whisper + AssemblyAI diarization": assign each Whisper segment to the
/// AssemblyAI speaker whose utterance overlaps it most. AssemblyAI's clean,
/// long speaker turns drive the labels; Whisper's higher-quality text is kept.
/// AAI's speaker labels ("A", "B", ...) are preserved so the rename modal works.
/// Whisper segment times are seconds; AAI utterance times are milliseconds.
pub fn assign_whisper_to_utterances(
    whisper_segments: &[(f64, f64, String)],
    utterances: &[Utterance],
) -> Vec<(f64, f64, String, String)> {
    whisper_segments
        .iter()
        .map(|(ws, we, text)| {
            let ws_ms = ws * 1000.0;
            let we_ms = we * 1000.0;
            let mut best = String::from("A");
            let mut best_overlap = 0.0_f64;
            for u in utterances {
                let overlap = (we_ms.min(u.end as f64) - ws_ms.max(u.start as f64)).max(0.0);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best = u.speaker.clone();
                }
            }
            (*ws, *we, text.clone(), best)
        })
        .collect()
}

/// AssemblyAI-shape JSON for the hybrid engine, so the speaker-rename modal works
/// uniformly. `segs` are (start_s, end_s, text, speaker_label).
pub fn merged_to_json(
    segs: &[(f64, f64, String, String)],
    duration: f64,
    whisper_model: &str,
) -> serde_json::Value {
    let utterances: Vec<serde_json::Value> = segs
        .iter()
        .map(|(start, end, text, speaker)| {
            serde_json::json!({
                "start": (start * 1000.0).round() as u64,
                "end": (end * 1000.0).round() as u64,
                "text": text.trim(),
                "speaker": speaker,
            })
        })
        .collect();

    serde_json::json!({
        // Text from local Whisper, speaker turns from AssemblyAI's diarizer.
        "engine": "whisper-assemblyai",
        "whisper_model": whisper_model,
        "audio_duration": duration,
        "utterances": utterances,
    })
}
