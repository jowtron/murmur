use crate::transcriber::{
    self, TranscriptionProgress, TranscriptionResult, WhisperModel,
};
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

        transcriber.transcribe(&audio_path, Some(progress_cb))
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
        vec![job.output_format.as_str()]
    };

    for fmt in &formats {
        let ext = *fmt;
        let out_path = output_dir.join(format!("{}.{}", stem, ext));
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

    let parsed: serde_json::Value =
        serde_json::from_str(content).map_err(|e| format!("Failed to parse chapters: {}", e))?;

    let chapters: Vec<Chapter> = serde_json::from_value(parsed["chapters"].clone())
        .map_err(|e| format!("Failed to parse chapter list: {}", e))?;

    Ok(chapters)
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
pub fn write_text_file(path: String, content: String) -> Result<(), String> {
    std::fs::write(&path, content).map_err(|e| format!("Failed to write file: {}", e))
}
