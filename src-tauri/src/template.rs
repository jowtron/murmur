use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Template {
    pub name: String,
    pub sample_rate: u32,
    pub samples: Vec<f32>,
    pub source_file: String,
    pub source_start: f64,
    pub source_end: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateInfo {
    pub name: String,
    pub duration_secs: f64,
    pub source_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateMatch {
    pub time_secs: f64,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaveformData {
    pub peaks_min: Vec<f32>,
    pub peaks_max: Vec<f32>,
    pub start_secs: f64,
    pub end_secs: f64,
    pub duration_secs: f64,
}

/// Directory for saved templates
pub fn templates_dir() -> PathBuf {
    let dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur")
        .join("templates");
    std::fs::create_dir_all(&dir).ok();
    dir
}

pub fn save_template(template: &Template) -> Result<(), String> {
    let dir = templates_dir();
    let filename = format!("{}.json", sanitize_name(&template.name));
    let path = dir.join(filename);
    let json = serde_json::to_string(template)
        .map_err(|e| format!("Failed to serialize template: {}", e))?;
    std::fs::write(&path, json)
        .map_err(|e| format!("Failed to write template: {}", e))?;
    Ok(())
}

pub fn load_template(name: &str) -> Result<Template, String> {
    let dir = templates_dir();
    let filename = format!("{}.json", sanitize_name(name));
    let path = dir.join(filename);
    let data = std::fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read template '{}': {}", name, e))?;
    serde_json::from_str(&data)
        .map_err(|e| format!("Failed to parse template '{}': {}", name, e))
}

pub fn list_templates() -> Vec<TemplateInfo> {
    let dir = templates_dir();
    let mut templates = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Ok(data) = std::fs::read_to_string(&path) {
                    if let Ok(t) = serde_json::from_str::<Template>(&data) {
                        templates.push(TemplateInfo {
                            name: t.name,
                            duration_secs: t.samples.len() as f64 / t.sample_rate as f64,
                            source_file: t.source_file,
                        });
                    }
                }
            }
        }
    }
    templates.sort_by(|a, b| a.name.cmp(&b.name));
    templates
}

pub fn delete_template(name: &str) -> Result<(), String> {
    let dir = templates_dir();
    let filename = format!("{}.json", sanitize_name(name));
    let path = dir.join(filename);
    std::fs::remove_file(&path)
        .map_err(|e| format!("Failed to delete template '{}': {}", name, e))?;
    Ok(())
}

/// Compute waveform peaks (min/max per bucket) for display
pub fn compute_waveform_peaks(
    samples: &[f32],
    sample_rate: u32,
    start_secs: f64,
    end_secs: f64,
    num_points: usize,
) -> WaveformData {
    let start_idx = (start_secs * sample_rate as f64) as usize;
    let end_idx = ((end_secs * sample_rate as f64) as usize).min(samples.len());

    if start_idx >= end_idx || num_points == 0 {
        return WaveformData {
            peaks_min: vec![],
            peaks_max: vec![],
            start_secs,
            end_secs,
            duration_secs: end_secs - start_secs,
        };
    }

    let region = &samples[start_idx..end_idx];
    let bucket_size = region.len() / num_points;
    let bucket_size = bucket_size.max(1);

    let mut peaks_min = Vec::with_capacity(num_points);
    let mut peaks_max = Vec::with_capacity(num_points);

    for i in 0..num_points {
        let start = i * bucket_size;
        let end = ((i + 1) * bucket_size).min(region.len());
        if start >= region.len() {
            break;
        }

        let chunk = &region[start..end];
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &s in chunk {
            if s < min { min = s; }
            if s > max { max = s; }
        }
        peaks_min.push(min);
        peaks_max.push(max);
    }

    WaveformData {
        peaks_min,
        peaks_max,
        start_secs,
        end_secs,
        duration_secs: end_secs - start_secs,
    }
}

/// Normalized cross-correlation to find template matches in audio.
/// Returns matches sorted by time, filtered by threshold.
pub fn find_matches(
    audio: &[f32],
    sample_rate: u32,
    template: &Template,
    threshold: f32,
    progress_cb: Option<Box<dyn FnMut(f32, &str) + Send>>,
) -> Vec<TemplateMatch> {
    let tmpl = &template.samples;
    let tmpl_len = tmpl.len();

    if audio.len() < tmpl_len || tmpl_len == 0 {
        return Vec::new();
    }

    // Mean-subtract template (Pearson-style correlation is more robust to volume/DC)
    let tmpl_mean: f64 = tmpl.iter().map(|&s| s as f64).sum::<f64>() / tmpl_len as f64;
    let tmpl_centered: Vec<f64> = tmpl.iter().map(|&s| s as f64 - tmpl_mean).collect();
    let tmpl_energy: f64 = tmpl_centered.iter().map(|v| v * v).sum();
    let tmpl_energy_sqrt = tmpl_energy.sqrt();

    if tmpl_energy_sqrt < 1e-10 {
        return Vec::new();
    }

    // Step size: check every 50ms worth of samples
    let step = (sample_rate as usize / 20).max(1);
    let total_positions = (audio.len() - tmpl_len) / step;
    let report_interval = (total_positions / 100).max(1);

    let mut progress_cb = progress_cb;
    let mut matches: Vec<TemplateMatch> = Vec::new();

    for (pos_idx, pos) in (0..audio.len() - tmpl_len).step_by(step).enumerate() {
        // Report progress
        if pos_idx % report_interval == 0 {
            if let Some(ref mut cb) = progress_cb {
                let pct = pos_idx as f32 / total_positions as f32;
                cb(pct, &format!("Scanning: {:.0}%", pct * 100.0));
            }
        }

        // Quick energy check — skip if window energy is very different
        // Only compute full correlation for windows with similar energy
        let window = &audio[pos..pos + tmpl_len];

        // Mean-subtract window then compute Pearson cross-correlation
        let mut win_mean: f64 = 0.0;
        for i in 0..tmpl_len { win_mean += window[i] as f64; }
        win_mean /= tmpl_len as f64;

        let mut cross: f64 = 0.0;
        let mut window_energy: f64 = 0.0;
        for i in 0..tmpl_len {
            let a = window[i] as f64 - win_mean;
            let b = tmpl_centered[i];
            cross += a * b;
            window_energy += a * a;
        }

        let window_energy_sqrt = window_energy.sqrt();
        if window_energy_sqrt < 1e-10 {
            continue;
        }

        let ncc = (cross / (tmpl_energy_sqrt * window_energy_sqrt)) as f32;

        if ncc >= threshold {
            let time = pos as f64 / sample_rate as f64;

            // Suppress nearby duplicates — keep the highest confidence within 1 second
            if let Some(last) = matches.last_mut() {
                if time - last.time_secs < 1.0 {
                    if ncc > last.confidence {
                        last.time_secs = time;
                        last.confidence = ncc;
                    }
                    continue;
                }
            }

            matches.push(TemplateMatch {
                time_secs: time,
                confidence: ncc,
            });
        }
    }

    if let Some(ref mut cb) = progress_cb {
        cb(1.0, &format!("Found {} matches", matches.len()));
    }

    matches
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}
