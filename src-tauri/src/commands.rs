use crate::flac_utils;
use crate::transcriber::{
    self, TranscriptionProgress, TranscriptionResult, WhisperModel,
};
use crate::gap_detection;
use crate::template;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{Emitter, State, Window};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

pub struct AppState {
    pub semaphore: Arc<Mutex<Arc<Semaphore>>>,
    pub cancel_tokens: Arc<Mutex<HashMap<String, CancellationToken>>>,
}

impl AppState {
    pub fn new(permits: usize) -> Self {
        Self {
            semaphore: Arc::new(Mutex::new(Arc::new(Semaphore::new(permits)))),
            cancel_tokens: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub display_name: String,
    pub downloaded: bool,
    pub filename: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobRequest {
    pub id: String,
    pub path: String,
    pub model: String,
    pub output_format: String,
    pub output_dir: Option<String>,
    pub threads: Option<i32>,
    #[serde(default)]
    pub per_word: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub gpu_cores: Option<u32>,
    pub metal_supported: bool,
    pub using_metal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterRequest {
    pub transcript: String,
    pub api_key: String,
    pub model: String,
    pub base_url: String,
    pub prompt: String,
    pub transcript_path: Option<String>,
    #[serde(default)]
    pub raw_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    pub title: String,
    pub start_time: String,
    pub start_secs: f64,
}

fn parse_model(name: &str) -> WhisperModel {
    match name {
        "tiny" => WhisperModel::Tiny,
        "base" => WhisperModel::Base,
        "small" => WhisperModel::Small,
        "medium" => WhisperModel::Medium,
        "large-v3" => WhisperModel::LargeV3,
        _ => WhisperModel::LargeV3Turbo,
    }
}

#[tauri::command]
pub fn list_models() -> Vec<ModelInfo> {
    let models = vec![
        WhisperModel::Tiny,
        WhisperModel::Base,
        WhisperModel::Small,
        WhisperModel::Medium,
        WhisperModel::LargeV3,
        WhisperModel::LargeV3Turbo,
    ];

    models
        .into_iter()
        .map(|m| {
            let path = transcriber::model_path(&m);
            let size_bytes = std::fs::metadata(&path).map(|meta| meta.len()).unwrap_or(0);
            ModelInfo {
                name: format!("{:?}", m)
                    .to_lowercase()
                    .replace("largev3turbo", "large-v3-turbo")
                    .replace("largev3", "large-v3"),
                display_name: m.display_name().to_string(),
                downloaded: transcriber::is_model_downloaded(&m),
                filename: m.filename().to_string(),
                size_bytes,
            }
        })
        .collect()
}

#[tauri::command]
pub fn is_model_ready(name: String) -> bool {
    let model = parse_model(&name);
    transcriber::is_model_downloaded(&model)
}

#[tauri::command]
pub async fn download_model(name: String, window: Window) -> Result<String, String> {
    let model = parse_model(&name);
    let url = model.url();
    let path = transcriber::model_path(&model);

    if path.exists() {
        return Ok("Already downloaded".to_string());
    }

    let tmp_path = path.with_extension("downloading");

    window
        .emit(
            "model-download-progress",
            serde_json::json!({"model": name, "progress": 0.0, "status": "downloading"}),
        )
        .ok();

    let mut child = tokio::process::Command::new("curl")
        .args([
            "-L",
            "-o",
            tmp_path.to_str().unwrap(),
            "--progress-bar",
            "--write-out",
            "%{size_download}",
            &url,
        ])
        .stderr(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    let win = window.clone();
    let model_name = name.clone();
    let tmp_path_clone = tmp_path.clone();
    let expected_sizes: HashMap<&str, u64> = [
        ("tiny", 77_700_000),
        ("base", 147_900_000),
        ("small", 487_600_000),
        ("medium", 1_533_700_000),
        ("large-v3", 3_094_400_000),
        ("large-v3-turbo", 1_627_800_000),
    ]
    .into();
    let expected = *expected_sizes.get(name.as_str()).unwrap_or(&500_000_000);

    let monitor = tokio::spawn(async move {
        loop {
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            if let Ok(meta) = tokio::fs::metadata(&tmp_path_clone).await {
                let progress = (meta.len() as f64 / expected as f64).min(0.99);
                let downloaded_mb = meta.len() as f64 / 1_048_576.0;
                let total_mb = expected as f64 / 1_048_576.0;
                win.emit(
                    "model-download-progress",
                    serde_json::json!({
                        "model": model_name,
                        "progress": progress,
                        "status": "downloading",
                        "downloaded_mb": downloaded_mb,
                        "total_mb": total_mb,
                    }),
                )
                .ok();
            }
        }
    });

    let status = child
        .wait()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    monitor.abort();

    if !status.success() {
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err("Download failed".to_string());
    }

    tokio::fs::rename(&tmp_path, &path)
        .await
        .map_err(|e| format!("Failed to finalize download: {}", e))?;

    window
        .emit(
            "model-download-progress",
            serde_json::json!({"model": name, "progress": 1.0, "status": "complete"}),
        )
        .ok();

    Ok("Downloaded".to_string())
}

#[tauri::command]
pub fn get_models_dir() -> String {
    transcriber::models_dir().to_string_lossy().to_string()
}

#[tauri::command]
pub fn get_gpu_info() -> GpuInfo {
    let output = std::process::Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output();

    let mut name = "Unknown".to_string();
    let mut gpu_cores = None;

    if let Ok(output) = output {
        if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&output.stdout) {
            if let Some(displays) = json["SPDisplaysDataType"].as_array() {
                if let Some(gpu) = displays.first() {
                    if let Some(n) = gpu["sppci_model"].as_str() {
                        name = n.to_string();
                    }
                    if let Some(cores) = gpu["sppci_cores"].as_str() {
                        gpu_cores = cores.parse().ok();
                    }
                }
            }
        }
    }

    GpuInfo {
        name,
        gpu_cores,
        metal_supported: true,
        using_metal: true,
    }
}

#[tauri::command]
pub fn set_concurrency(permits: usize, state: State<'_, AppState>) {
    let new_sem = Arc::new(Semaphore::new(permits));
    let mut sem = state.semaphore.lock().unwrap();
    *sem = new_sem;
}

#[tauri::command]
pub fn cancel_job(job_id: String, state: State<'_, AppState>) {
    let tokens = state.cancel_tokens.lock().unwrap();
    if let Some(token) = tokens.get(&job_id) {
        token.cancel();
    }
}

#[tauri::command]
pub async fn transcribe_file(
    job: JobRequest,
    window: Window,
    state: State<'_, AppState>,
) -> Result<TranscriptionResult, String> {
    let model = parse_model(&job.model);

    if !transcriber::is_model_downloaded(&model) {
        return Err(format!(
            "Model '{}' is not downloaded. Please download it from the Models manager first.",
            job.model
        ));
    }

    // Create cancellation token for this job
    let cancel_token = CancellationToken::new();
    {
        let mut tokens = state.cancel_tokens.lock().unwrap();
        tokens.insert(job.id.clone(), cancel_token.clone());
    }

    let sem = state.semaphore.lock().unwrap().clone();
    let permit = sem
        .acquire_owned()
        .await
        .map_err(|e| format!("Queue error: {}", e))?;

    // Check if cancelled while waiting
    if cancel_token.is_cancelled() {
        drop(permit);
        return Err("Cancelled".to_string());
    }

    let audio_path = PathBuf::from(&job.path);
    let job_id = job.id.clone();
    let file_name = audio_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    let threads = job.threads.unwrap_or(8);

    window
        .emit(
            "transcription-progress",
            TranscriptionProgress {
                job_id: job_id.clone(),
                file: file_name.clone(),
                progress: 0.0,
                status: "loading_model".to_string(),
            },
        )
        .ok();

    let win = window.clone();
    let jid = job_id.clone();
    let fname = file_name.clone();
    let ct = cancel_token.clone();

    let result = tokio::task::spawn_blocking(move || {
        let transcriber = transcriber::Transcriber::new(&model, threads)?;

        let progress_cb: Arc<Mutex<dyn FnMut(f32) + Send>> = Arc::new(Mutex::new({
            let win = win.clone();
            let jid = jid.clone();
            let fname = fname.clone();
            let ct = ct.clone();
            move |progress: f32| {
                if ct.is_cancelled() {
                    return;
                }
                win.emit(
                    "transcription-progress",
                    TranscriptionProgress {
                        job_id: jid.clone(),
                        file: fname.clone(),
                        progress,
                        status: "transcribing".to_string(),
                    },
                )
                .ok();
            }
        }));

        if job.per_word {
            transcriber.transcribe_per_word(&audio_path, Some(progress_cb))
        } else {
            transcriber.transcribe(&audio_path, Some(progress_cb))
        }
    })
    .await
    .map_err(|e| format!("Task error: {}", e))??;

    // Check if cancelled during transcription
    if cancel_token.is_cancelled() {
        // Clean up cancel token
        state.cancel_tokens.lock().unwrap().remove(&job.id);
        drop(permit);
        return Err("Cancelled".to_string());
    }

    // Save output
    let output_dir = job
        .output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&job.path).parent().unwrap().to_path_buf());

    let stem = PathBuf::from(&job.path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let formats: Vec<&str> = if job.output_format == "all" {
        vec!["txt", "srt", "vtt", "json"]
    } else {
        job.output_format.split(',').map(|s| s.trim()).collect()
    };

    for fmt in &formats {
        let ext = *fmt;
        let out_path = output_dir.join(format!("{}_transcription_{}.{}", stem, job.model, ext));
        let content = match ext {
            "srt" => transcriber::to_srt(&result),
            "vtt" => transcriber::to_vtt(&result),
            "json" => serde_json::to_string_pretty(&result).unwrap_or_default(),
            _ => result.text.clone(),
        };
        std::fs::write(&out_path, content)
            .map_err(|e| format!("Failed to write {}: {}", out_path.display(), e))?;
    }

    window
        .emit(
            "transcription-progress",
            TranscriptionProgress {
                job_id: job_id.clone(),
                file: file_name,
                progress: 1.0,
                status: "complete".to_string(),
            },
        )
        .ok();

    // Clean up cancel token
    state.cancel_tokens.lock().unwrap().remove(&job.id);
    drop(permit);
    Ok(result)
}

/// Condense SRT content to just "[HH:MM:SS] text" lines to reduce token count
fn condense_srt(input: &str) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        // Look for timestamp lines like "00:00:00,000 --> 00:00:05,000"
        if line.contains("-->") {
            let time = line.split("-->").next().unwrap_or("").trim();
            // Extract just HH:MM:SS from "HH:MM:SS,mmm"
            let time_short = time.split(',').next().unwrap_or(time).trim();
            // Collect text lines until empty line
            i += 1;
            let mut text = String::new();
            while i < lines.len() && !lines[i].trim().is_empty() {
                if !text.is_empty() { text.push(' '); }
                text.push_str(lines[i].trim());
                i += 1;
            }
            if !text.is_empty() {
                result.push_str(&format!("[{}] {}\n", time_short, text));
            }
        }
        i += 1;
    }
    if result.is_empty() {
        // Not SRT format, return as-is
        input.to_string()
    } else {
        result
    }
}

#[tauri::command]
pub async fn detect_chapters(req: ChapterRequest) -> Result<Vec<Chapter>, String> {
    // Check LLM cache
    let cache_path = req.transcript_path.as_ref().map(|p| {
        let path = PathBuf::from(p);
        let raw_stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
        // Strip _transcription or _transcription_<model> suffix
        let stem = if let Some(idx) = raw_stem.find("_transcription") {
            raw_stem[..idx].to_string()
        } else {
            raw_stem
        };
        let dir = path.parent().unwrap_or(std::path::Path::new("."));
        let llm_short = req.model.split('/').last().unwrap_or(&req.model);
        let suffix = if req.raw_mode { "_llm_raw" } else { "_llm_chapters" };
        dir.join(format!("{}_{}{}.json", stem, llm_short, suffix))
    });

    if let Some(ref cp) = cache_path {
        if cp.exists() {
            if let Ok(data) = std::fs::read_to_string(cp) {
                if let Ok(chapters) = serde_json::from_str::<Vec<Chapter>>(&data) {
                    return Ok(chapters);
                }
            }
        }
    }

    // Condense SRT to reduce token usage
    let transcript = condense_srt(&req.transcript);

    let body = serde_json::json!({
        "model": req.model,
        "messages": [
            {
                "role": "system",
                "content": req.prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "chapters",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {
                        "chapters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "title": { "type": "string" },
                                    "start_time": { "type": "string" },
                                    "start_secs": { "type": "number" }
                                },
                                "required": ["title", "start_time", "start_secs"],
                                "additionalProperties": false
                            }
                        }
                    },
                    "required": ["chapters"],
                    "additionalProperties": false
                }
            }
        }
    });

    let output = tokio::process::Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            &req.base_url,
            "-H",
            "Content-Type: application/json",
            "-H",
            &format!("Authorization: Bearer {}", req.api_key),
            "-H",
            "HTTP-Referer: https://whisper-transcriber.app",
            "-H",
            "X-Title: Whisper_Transcriber",
            "-d",
            &body.to_string(),
        ])
        .output()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !output.status.success() {
        return Err("API request failed".to_string());
    }

    let response: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|e| format!("Failed to parse response: {}", e))?;

    let content = response["choices"][0]["message"]["content"]
        .as_str()
        .ok_or_else(|| {
            let err = response["error"]["message"]
                .as_str()
                .unwrap_or("Unknown API error");
            format!("API error: {}", err)
        })?;

    // Strip markdown code fences if present (e.g. ```json ... ```)
    let clean_content = content.trim();
    let clean_content = if clean_content.starts_with("```") {
        let inner = clean_content
            .strip_prefix("```json")
            .or_else(|| clean_content.strip_prefix("```"))
            .unwrap_or(clean_content);
        inner.strip_suffix("```").unwrap_or(inner).trim()
    } else {
        clean_content
    };

    let parsed: serde_json::Value =
        serde_json::from_str(clean_content).map_err(|e| format!("Failed to parse chapters: {}", e))?;

    let chapters: Vec<Chapter> = serde_json::from_value(parsed["chapters"].clone())
        .map_err(|e| format!("Failed to parse chapter list: {}", e))?;

    // Always save raw LLM output alongside (even in corrected mode)
    if !req.raw_mode {
        if let Some(ref cp) = cache_path {
            let raw_path = cp.with_file_name(
                cp.file_name().unwrap().to_string_lossy().replace("_llm_chapters.", "_llm_raw."),
            );
            if let Ok(json) = serde_json::to_string_pretty(&chapters) {
                std::fs::write(&raw_path, json).ok();
            }
        }
    }

    let final_chapters = if req.raw_mode {
        chapters
    } else {
        correct_chapter_timestamps(&chapters, &req.transcript)
    };

    // Save (possibly corrected) cache
    if let Some(ref cp) = cache_path {
        if let Ok(json) = serde_json::to_string_pretty(&final_chapters) {
            std::fs::write(cp, json).ok();
        }
    }

    Ok(final_chapters)
}

/// Search the original SRT transcript for each chapter title and use
/// the transcript's own timestamp instead of the LLM's (often inaccurate) one.
fn correct_chapter_timestamps(chapters: &[Chapter], transcript: &str) -> Vec<Chapter> {
    // Parse SRT into (start_secs, text) entries
    let entries = parse_srt_entries(transcript);
    if entries.is_empty() {
        return chapters.to_vec();
    }

    const ANCHOR_WINDOW: f64 = 60.0; // search ±60s around the LLM's time
    let result: Vec<Chapter> = chapters.iter().enumerate().map(|(idx, ch)| {
        if idx == 0 {
            return Chapter {
                title: ch.title.clone(),
                start_time: "00:00:00".to_string(),
                start_secs: 0.0,
            };
        }

        let title_lower = ch.title.to_lowercase();
        let title_words: Vec<&str> = title_lower.split_whitespace()
            .filter(|w| w.len() > 2)
            .collect();

        // Search only within ±ANCHOR_WINDOW of the LLM's claimed time
        let anchor = ch.start_secs;
        let lo = (anchor - ANCHOR_WINDOW).max(0.0);
        let hi = anchor + ANCHOR_WINDOW;
        let candidates: Vec<&(f64, String)> = entries.iter()
            .filter(|(s, _)| *s >= lo && *s <= hi)
            .collect();

        // Try progressively shorter n-word phrases, longest first; pick the match
        // closest to the anchor time
        let mut found: Option<f64> = None;
        if title_words.len() >= 1 {
            'outer: for n in (1..=title_words.len()).rev() {
                let mut best: Option<f64> = None;
                let mut best_dist = f64::MAX;
                for start in 0..=(title_words.len() - n) {
                    let phrase = title_words[start..start + n].join(" ");
                    if phrase.len() < 5 { continue; }
                    for (s, text) in candidates.iter() {
                        if text.to_lowercase().contains(&phrase) {
                            let dist = (*s - anchor).abs();
                            if dist < best_dist {
                                best_dist = dist;
                                best = Some(*s);
                            }
                        }
                    }
                }
                if best.is_some() { found = best; break 'outer; }
            }
        }

        // If no match in window, keep LLM's time
        let secs = found.unwrap_or(ch.start_secs);
        let h = (secs / 3600.0) as u32;
        let m = ((secs % 3600.0) / 60.0) as u32;
        let s = (secs % 60.0) as u32;
        Chapter {
            title: ch.title.clone(),
            start_time: format!("{:02}:{:02}:{:02}", h, m, s),
            start_secs: secs,
        }
    }).collect();
    result
}

/// Parse SRT format into (start_seconds, text) pairs
fn parse_srt_entries(input: &str) -> Vec<(f64, String)> {
    let mut entries = Vec::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i].trim();
        if line.contains("-->") {
            let time_str = line.split("-->").next().unwrap_or("").trim();
            let secs = parse_srt_time(time_str);
            i += 1;
            let mut text = String::new();
            while i < lines.len() && !lines[i].trim().is_empty() {
                if !text.is_empty() { text.push(' '); }
                text.push_str(lines[i].trim());
                i += 1;
            }
            if !text.is_empty() {
                entries.push((secs, text));
            }
        }
        i += 1;
    }
    entries
}

/// Parse "HH:MM:SS,mmm" or "HH:MM:SS.mmm" to seconds
fn parse_srt_time(s: &str) -> f64 {
    let s = s.replace(',', ".");
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() == 3 {
        let h: f64 = parts[0].parse().unwrap_or(0.0);
        let m: f64 = parts[1].parse().unwrap_or(0.0);
        let sec: f64 = parts[2].parse().unwrap_or(0.0);
        h * 3600.0 + m * 60.0 + sec
    } else {
        0.0
    }
}

#[tauri::command]
pub async fn scan_directory(path: String) -> Result<Vec<String>, String> {
    let dir = PathBuf::from(&path);
    if !dir.is_dir() {
        return Err("Not a directory".to_string());
    }

    let extensions = ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus"];
    let mut files = Vec::new();

    fn walk(dir: &PathBuf, extensions: &[&str], files: &mut Vec<String>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk(&path, extensions, files);
                } else if let Some(ext) = path.extension() {
                    if extensions.contains(&ext.to_string_lossy().to_lowercase().as_str()) {
                        files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }
    }

    walk(&dir, &extensions, &mut files);
    files.sort();
    Ok(files)
}

#[tauri::command]
pub fn get_audio_duration(path: String) -> Result<f64, String> {
    transcriber::get_duration(&PathBuf::from(path))
}

#[tauri::command]
pub fn read_text_file(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| format!("Failed to read file: {}", e))
}

#[tauri::command]
pub fn file_exists(path: String) -> bool {
    std::path::Path::new(&path).exists()
}

#[tauri::command]
pub fn write_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))
}

#[tauri::command]
pub async fn detect_energy_gaps(
    path: String,
    min_gap_secs: f64,
    silence_threshold: f32,
    window: Window,
) -> Result<Vec<gap_detection::SpeechGap>, String> {
    let audio_path = PathBuf::from(&path);

    // Check cache first
    if let Some(cached) = gap_detection::load_cached_gaps(&audio_path) {
        window.emit("gap-progress", serde_json::json!({"status": "Using cached gap results", "progress": 1.0})).ok();
        return Ok(cached);
    }

    let win = window.clone();
    let gaps = tokio::task::spawn_blocking(move || {
        let progress_cb: Arc<Mutex<dyn FnMut(f32, &str) + Send>> = Arc::new(Mutex::new({
            let win = win.clone();
            move |progress: f32, status: &str| {
                win.emit("gap-progress", serde_json::json!({"status": status, "progress": progress})).ok();
            }
        }));

        let samples = transcriber::audio_to_pcm(std::path::Path::new(&path))?;
        Ok::<Vec<gap_detection::SpeechGap>, String>(gap_detection::detect_energy_gaps(
            &samples, 16000, 0.5, silence_threshold, min_gap_secs, Some(progress_cb),
        ))
    })
    .await
    .map_err(|e| format!("Task error: {}", e))??;

    // Save cache
    gap_detection::save_cached_gaps(&audio_path, &gaps).ok();

    Ok(gaps)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterWithSnap {
    pub title: String,
    pub start_time: String,
    pub start_secs: f64,
    pub original_secs: f64,
    pub snapped: bool,
}

#[tauri::command]
pub async fn detect_chapters_with_gaps(
    req: ChapterRequest,
    audio_path: String,
    min_gap_secs: f64,
    silence_threshold: f32,
    max_lookback_secs: f64,
    window: Window,
) -> Result<Vec<ChapterWithSnap>, String> {
    // Step 1: LLM chapters first
    window.emit("gap-progress", serde_json::json!({"status": "Sending transcript to LLM...", "progress": 0.0})).ok();
    let chapters = detect_chapters(req).await?;

    if chapters.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: Decode audio and scan only 30s windows before each chapter
    let win = window.clone();
    let ap = audio_path.clone();
    let chapter_times: Vec<f64> = chapters.iter().map(|c| c.start_secs).collect();
    let lookback = max_lookback_secs;
    let thresh = silence_threshold;
    let min_gap = min_gap_secs;

    let snapped = tokio::task::spawn_blocking(move || {
        win.emit("gap-progress", serde_json::json!({"status": "Decoding audio...", "progress": 0.1})).ok();
        let samples = transcriber::audio_to_pcm(std::path::Path::new(&ap))?;
        let sample_rate = 16000u32;

        let mut results: Vec<Option<f64>> = Vec::new();

        for (i, &chapter_time) in chapter_times.iter().enumerate() {
            let pct = 0.2 + 0.8 * (i as f32 / chapter_times.len() as f32);
            win.emit("gap-progress", serde_json::json!({
                "status": format!("Scanning gap for chapter {}/{}", i + 1, chapter_times.len()),
                "progress": pct,
            })).ok();

            // Extract window: lookback seconds before chapter_time
            let end_sample = (chapter_time * sample_rate as f64) as usize;
            let start_sample = ((chapter_time - lookback).max(0.0) * sample_rate as f64) as usize;

            if start_sample >= samples.len() || end_sample > samples.len() {
                results.push(None);
                continue;
            }

            let window_samples = &samples[start_sample..end_sample.min(samples.len())];

            // Find gaps in this window
            let gaps = gap_detection::detect_energy_gaps(
                window_samples, sample_rate, 0.5, thresh, min_gap, None,
            );

            // Adjust gap times back to absolute time
            let window_offset = start_sample as f64 / sample_rate as f64;

            // Find the gap closest to (but before) chapter_time
            let best = gaps.iter()
                .map(|g| gap_detection::SpeechGap {
                    start_secs: g.start_secs + window_offset,
                    end_secs: g.end_secs + window_offset,
                    duration_secs: g.duration_secs,
                })
                .filter(|g| g.start_secs <= chapter_time)
                .max_by(|a, b| a.start_secs.partial_cmp(&b.start_secs).unwrap())
                .map(|g| g.start_secs);

            results.push(best);
        }

        win.emit("gap-progress", serde_json::json!({"status": "Done", "progress": 1.0})).ok();
        Ok::<Vec<Option<f64>>, String>(results)
    })
    .await
    .map_err(|e| format!("Task error: {}", e))??;

    let result = chapters
        .iter()
        .zip(snapped.iter())
        .map(|(ch, snapped_time)| {
            let (final_secs, was_snapped) = match snapped_time {
                Some(t) => (*t, true),
                None => (ch.start_secs, false),
            };
            let h = (final_secs / 3600.0) as u32;
            let m = ((final_secs % 3600.0) / 60.0) as u32;
            let s = (final_secs % 60.0) as u32;
            ChapterWithSnap {
                title: ch.title.clone(),
                start_time: format!("{:02}:{:02}:{:02}", h, m, s),
                start_secs: final_secs,
                original_secs: ch.start_secs,
                snapped: was_snapped,
            }
        })
        .collect();

    Ok(result)
}

#[tauri::command]
pub fn check_transcription_exists(path: String, model: String, output_dir: Option<String>) -> bool {
    let audio_path = PathBuf::from(&path);
    let dir = output_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| audio_path.parent().unwrap_or(std::path::Path::new(".")).to_path_buf());
    let stem = audio_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    // Check if SRT exists (the primary format we need for chapter detection)
    let srt_path = dir.join(format!("{}_transcription_{}.srt", stem, model));
    srt_path.exists()
}

#[tauri::command]
pub fn check_seekability(path: String) -> flac_utils::SeekInfo {
    flac_utils::check_seekability(std::path::Path::new(&path))
}

#[tauri::command]
pub async fn fix_seektable(path: String) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        flac_utils::add_seektable(std::path::Path::new(&path))?;
        Ok("Seek table added".to_string())
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}

#[tauri::command]
pub async fn fix_seektables_batch(paths: Vec<String>) -> Result<Vec<String>, String> {
    tokio::task::spawn_blocking(move || {
        let mut fixed = Vec::new();
        for path in &paths {
            let p = std::path::Path::new(path);
            if let Ok(false) = flac_utils::has_seektable(p) {
                if flac_utils::add_seektable(p).is_ok() {
                    fixed.push(path.clone());
                }
            }
        }
        Ok(fixed)
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedChaptersRequest {
    pub audio_path: String,
    pub chapters: Vec<Chapter>,
}

#[tauri::command]
pub async fn embed_chapters_in_flac(req: EmbedChaptersRequest) -> Result<String, String> {
    let path = PathBuf::from(&req.audio_path);
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    let chapters: Vec<(String, f64)> = req.chapters.iter()
        .map(|c| (c.title.clone(), c.start_secs))
        .collect();

    if ext == "flac" {
        tokio::task::spawn_blocking(move || {
            flac_utils::embed_chapters(&path, &chapters)
        })
        .await
        .map_err(|e| format!("Task error: {}", e))??;

        Ok("Chapters embedded in FLAC".to_string())
    } else {
        Err(format!("Chapter embedding not supported for .{} files", ext))
    }
}

#[tauri::command]
pub async fn write_cue_file(audio_path: String, chapters: Vec<Chapter>) -> Result<String, String> {
    let path = PathBuf::from(&audio_path);
    let chapter_data: Vec<(String, f64)> = chapters.iter()
        .map(|c| (c.title.clone(), c.start_secs))
        .collect();

    let cue = flac_utils::generate_cue(&path, &chapter_data)?;

    let cue_path = path.with_extension("cue");
    std::fs::write(&cue_path, &cue)
        .map_err(|e| format!("Failed to write cue file: {}", e))?;

    Ok(cue_path.to_string_lossy().to_string())
}

// === Template matching commands ===

static PCM_CACHE: std::sync::Mutex<Option<(String, std::sync::Arc<Vec<f32>>)>> = std::sync::Mutex::new(None);

fn get_cached_pcm(path: &str) -> Result<std::sync::Arc<Vec<f32>>, String> {
    {
        let guard = PCM_CACHE.lock().unwrap();
        if let Some((p, samples)) = guard.as_ref() {
            if p == path {
                return Ok(samples.clone());
            }
        }
    }
    let samples = std::sync::Arc::new(transcriber::audio_to_pcm(std::path::Path::new(path))?);
    *PCM_CACHE.lock().unwrap() = Some((path.to_string(), samples.clone()));
    Ok(samples)
}

#[tauri::command]
pub async fn get_waveform_peaks(
    path: String,
    start_secs: f64,
    end_secs: f64,
    num_points: usize,
) -> Result<template::WaveformData, String> {
    tokio::task::spawn_blocking(move || {
        let samples = get_cached_pcm(&path)?;
        let sample_rate = 16000u32;
        let actual_end = if end_secs <= 0.0 {
            samples.len() as f64 / sample_rate as f64
        } else {
            end_secs
        };
        Ok(template::compute_waveform_peaks(
            &samples, sample_rate, start_secs, actual_end, num_points,
        ))
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}

#[tauri::command]
pub async fn get_audio_region_pcm(
    path: String,
    start_secs: f64,
    end_secs: f64,
) -> Result<Vec<f32>, String> {
    tokio::task::spawn_blocking(move || {
        let samples = get_cached_pcm(&path)?;
        let sample_rate = 16000u32;
        let start_idx = (start_secs * sample_rate as f64) as usize;
        let end_idx = ((end_secs * sample_rate as f64) as usize).min(samples.len());
        if start_idx >= end_idx {
            return Ok(Vec::new());
        }
        Ok(samples[start_idx..end_idx].to_vec())
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}

#[tauri::command]
pub async fn save_audio_template(
    name: String,
    path: String,
    start_secs: f64,
    end_secs: f64,
) -> Result<String, String> {
    tokio::task::spawn_blocking(move || {
        let samples = get_cached_pcm(&path)?;
        let sample_rate = 16000u32;
        let start_idx = (start_secs * sample_rate as f64) as usize;
        let end_idx = ((end_secs * sample_rate as f64) as usize).min(samples.len());
        if start_idx >= end_idx {
            return Err("Invalid time range".to_string());
        }
        let region = samples[start_idx..end_idx].to_vec();
        let t = template::Template {
            name: name.clone(),
            sample_rate,
            samples: region,
            source_file: path,
            source_start: start_secs,
            source_end: end_secs,
        };
        template::save_template(&t)?;
        Ok(format!("Template '{}' saved ({:.1}s)", name, end_secs - start_secs))
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}

#[tauri::command]
pub fn list_audio_templates() -> Vec<template::TemplateInfo> {
    template::list_templates()
}

#[tauri::command]
pub fn get_template_pcm(name: String) -> Result<Vec<f32>, String> {
    let t = template::load_template(&name)?;
    Ok(t.samples)
}

#[tauri::command]
pub fn delete_audio_template(name: String) -> Result<(), String> {
    template::delete_template(&name)
}

#[tauri::command]
pub async fn find_template_matches(
    audio_path: String,
    template_name: String,
    threshold: f32,
    window: Window,
) -> Result<Vec<template::TemplateMatch>, String> {
    let tmpl = template::load_template(&template_name)?;

    tokio::task::spawn_blocking(move || {
        let samples = transcriber::audio_to_pcm(std::path::Path::new(&audio_path))?;
        let sample_rate = 16000u32;

        let win = window.clone();
        let progress_cb = Box::new(move |progress: f32, status: &str| {
            win.emit("template-match-progress", serde_json::json!({
                "progress": progress,
                "status": status,
            })).ok();
        });

        let matches = template::find_matches(
            &samples, sample_rate, &tmpl, threshold,
            Some(progress_cb),
        );
        Ok(matches)
    })
    .await
    .map_err(|e| format!("Task error: {}", e))?
}
