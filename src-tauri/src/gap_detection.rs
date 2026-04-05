use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// A detected gap (quiet region) in the audio
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SpeechGap {
    pub start_secs: f64,
    pub end_secs: f64,
    pub duration_secs: f64,
}

/// Cache path for gap detection: same dir as audio, stem.gaps.json
pub fn gap_cache_path(audio_path: &Path) -> PathBuf {
    let stem = audio_path.file_stem().unwrap_or_default().to_string_lossy();
    let dir = audio_path.parent().unwrap_or(Path::new("."));
    dir.join(format!("{}.gaps.json", stem))
}

/// Load cached gaps if the cache file exists and is newer than the audio file
pub fn load_cached_gaps(audio_path: &Path) -> Option<Vec<SpeechGap>> {
    let cache = gap_cache_path(audio_path);
    if !cache.exists() {
        return None;
    }
    let audio_modified = std::fs::metadata(audio_path).ok()?.modified().ok()?;
    let cache_modified = std::fs::metadata(&cache).ok()?.modified().ok()?;
    if cache_modified < audio_modified {
        return None;
    }
    let data = std::fs::read_to_string(&cache).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save gaps to cache file
pub fn save_cached_gaps(audio_path: &Path, gaps: &[SpeechGap]) -> Result<(), String> {
    let cache = gap_cache_path(audio_path);
    let json = serde_json::to_string_pretty(gaps)
        .map_err(|e| format!("Failed to serialize gap cache: {}", e))?;
    std::fs::write(&cache, json)
        .map_err(|e| format!("Failed to write gap cache: {}", e))?;
    Ok(())
}

/// Detect quiet gaps in audio using RMS energy analysis.
/// This works much better than Silero VAD for produced audiobooks with music beds.
///
/// - `rms_window_secs`: window size for RMS calculation (e.g. 0.5s)
/// - `silence_threshold`: RMS below this is considered quiet (e.g. 0.02)
/// - `min_gap_secs`: minimum gap duration to report (e.g. 1.5s)
/// - `progress_cb`: optional callback for (progress 0-1, status message)
pub fn detect_energy_gaps(
    samples: &[f32],
    sample_rate: u32,
    rms_window_secs: f64,
    silence_threshold: f32,
    min_gap_secs: f64,
    progress_cb: Option<Arc<Mutex<dyn FnMut(f32, &str) + Send>>>,
) -> Vec<SpeechGap> {
    let window_samples = (rms_window_secs * sample_rate as f64) as usize;
    if window_samples == 0 || samples.len() < window_samples {
        return Vec::new();
    }

    let total_windows = samples.len() / window_samples;
    let report_interval = (total_windows / 50).max(1);

    if let Some(ref cb) = progress_cb {
        if let Ok(mut f) = cb.lock() {
            let duration = samples.len() as f64 / sample_rate as f64;
            f(0.01, &format!("Analyzing {:.0}s of audio for quiet gaps...", duration));
        }
    }

    // Calculate RMS energy for each window
    let mut rms_values: Vec<(f64, f32)> = Vec::with_capacity(total_windows);
    for i in 0..total_windows {
        let start = i * window_samples;
        let chunk = &samples[start..start + window_samples];

        let sum_sq: f32 = chunk.iter().map(|s| s * s).sum();
        let rms = (sum_sq / chunk.len() as f32).sqrt();
        let time = start as f64 / sample_rate as f64;
        rms_values.push((time, rms));

        if i % report_interval == 0 {
            if let Some(ref cb) = progress_cb {
                if let Ok(mut f) = cb.lock() {
                    let pct = i as f32 / total_windows as f32;
                    f(pct, &format!("Energy analysis: {:.0}% ({:.0}s / {:.0}s)",
                        pct * 100.0, time,
                        samples.len() as f64 / sample_rate as f64));
                }
            }
        }
    }

    // Find consecutive quiet windows
    let mut gaps = Vec::new();
    let mut gap_start: Option<f64> = None;

    for &(time, rms) in &rms_values {
        if rms < silence_threshold {
            if gap_start.is_none() {
                gap_start = Some(time);
            }
        } else {
            if let Some(start) = gap_start {
                let duration = time - start;
                if duration >= min_gap_secs {
                    gaps.push(SpeechGap {
                        start_secs: start,
                        end_secs: time,
                        duration_secs: duration,
                    });
                }
                gap_start = None;
            }
        }
    }

    // Handle trailing gap
    if let Some(start) = gap_start {
        let end = samples.len() as f64 / sample_rate as f64;
        let duration = end - start;
        if duration >= min_gap_secs {
            gaps.push(SpeechGap {
                start_secs: start,
                end_secs: end,
                duration_secs: duration,
            });
        }
    }

    if let Some(ref cb) = progress_cb {
        if let Ok(mut f) = cb.lock() {
            f(1.0, &format!("Found {} quiet gaps", gaps.len()));
        }
    }

    gaps
}

