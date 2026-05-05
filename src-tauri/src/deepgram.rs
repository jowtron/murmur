use serde::{Deserialize, Serialize};
use std::path::Path;

const API_BASE: &str = "https://api.deepgram.com/v1/listen";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: Option<f64>,
    pub speaker: Option<u32>,
    pub punctuated_word: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    pub start: f64,
    pub end: f64,
    pub transcript: String,
    pub speaker: u32,
    #[serde(default)]
    pub words: Vec<Word>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub duration: Option<f64>,
    pub channels: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepgramResponse {
    pub metadata: Option<Metadata>,
    pub results: Option<Results>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Results {
    #[serde(default)]
    pub utterances: Option<Vec<Utterance>>,
}

fn content_type_for(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase());
    match ext.as_deref() {
        Some("mp3") => "audio/mpeg",
        Some("m4a") | Some("aac") => "audio/mp4",
        Some("flac") => "audio/flac",
        Some("ogg") | Some("opus") => "audio/ogg",
        Some("wav") => "audio/wav",
        Some("wma") => "audio/x-ms-wma",
        _ => "audio/*",
    }
}

/// Submit a local file synchronously to Deepgram. Returns the parsed response.
pub fn transcribe(
    path: &Path,
    api_key: &str,
    model: &str,
    language_code: Option<&str>,
) -> Result<DeepgramResponse, String> {
    let path_str = path
        .to_str()
        .ok_or_else(|| "Path is not valid UTF-8".to_string())?;
    let auth = format!("Authorization: Token {}", api_key);
    let ct = format!("Content-Type: {}", content_type_for(path));
    let data_arg = format!("@{}", path_str);

    // Build query string
    let mut params = vec![
        format!("model={}", urlencode(model)),
        "smart_format=true".to_string(),
        "punctuate=true".to_string(),
        "diarize=true".to_string(),
        "utterances=true".to_string(),
    ];
    if let Some(lc) = language_code {
        params.push(format!("language={}", urlencode(lc)));
    }
    let url = format!("{}?{}", API_BASE, params.join("&"));

    let output = std::process::Command::new("curl")
        .args([
            "-sS",
            "--fail-with-body",
            "-X", "POST",
            "-H", &auth,
            "-H", &ct,
            "--data-binary", &data_arg,
            &url,
        ])
        .output()
        .map_err(|e| format!("curl failed to start: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Deepgram request failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let body = output.stdout;
    serde_json::from_slice(&body)
        .map_err(|e| format!("Failed to parse Deepgram response: {} | body: {}", e, String::from_utf8_lossy(&body)))
}

fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.bytes() {
        match c {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(c as char);
            }
            _ => out.push_str(&format!("%{:02X}", c)),
        }
    }
    out
}

fn fmt_srt_time(secs: f64) -> String {
    let total_ms = (secs * 1000.0).round() as u64;
    let total_secs = total_ms / 1000;
    let h = total_secs / 3600;
    let m = (total_secs % 3600) / 60;
    let s = total_secs % 60;
    let mmm = total_ms % 1000;
    format!("{:02}:{:02}:{:02},{:03}", h, m, s, mmm)
}

/// Speaker `0`/`1` -> `Speaker A`/`Speaker B`.
fn speaker_label(n: u32) -> String {
    let c = (b'A' + (n as u8 % 26)) as char;
    c.to_string()
}

pub fn utterances_to_srt(utts: &[Utterance]) -> String {
    let mut out = String::new();
    for (i, u) in utts.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", fmt_srt_time(u.start), fmt_srt_time(u.end)));
        out.push_str(&format!("Speaker {}: {}\n\n", speaker_label(u.speaker), u.transcript.trim()));
    }
    out
}

pub fn utterances_to_text(utts: &[Utterance]) -> String {
    let mut out = String::new();
    for u in utts {
        out.push_str(&format!("Speaker {}: {}\n\n", speaker_label(u.speaker), u.transcript.trim()));
    }
    out
}

/// Convert numeric speaker IDs to letter labels in a synthetic AssemblyAI-style utterance list,
/// which is what the speaker rename modal expects to consume from the JSON file.
pub fn to_aai_style_utterances(utts: &[Utterance]) -> Vec<serde_json::Value> {
    utts.iter()
        .map(|u| {
            serde_json::json!({
                "start": (u.start * 1000.0).round() as u64,
                "end": (u.end * 1000.0).round() as u64,
                "text": u.transcript.trim(),
                "speaker": speaker_label(u.speaker),
                "words": u.words.iter().map(|w| serde_json::json!({
                    "start": (w.start * 1000.0).round() as u64,
                    "end": (w.end * 1000.0).round() as u64,
                    "text": w.punctuated_word.clone().unwrap_or_else(|| w.word.clone()),
                    "speaker": w.speaker.map(speaker_label),
                })).collect::<Vec<_>>(),
            })
        })
        .collect()
}
