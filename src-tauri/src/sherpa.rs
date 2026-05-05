use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub speaker: i32,
}

pub fn models_dir() -> PathBuf {
    let dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur")
        .join("sherpa-models");
    std::fs::create_dir_all(&dir).ok();
    dir
}

pub fn segmentation_path() -> PathBuf {
    models_dir().join("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
}

pub fn embedding_path() -> PathBuf {
    models_dir().join("nemo_en_speakerverification_speakernet.onnx")
}

pub fn models_ready() -> bool {
    segmentation_path().exists() && embedding_path().exists()
}

/// Run speaker diarization on 16kHz mono f32 PCM.
/// Returns segments sorted by start time.
///
/// `num_speakers` — if > 0, force clustering to this exact count (use when you know
///   how many speakers are in the file). If 0, the threshold is used to auto-cluster.
/// `threshold` — distance threshold for the FastClustering algorithm. With the NeMo
///   SpeakerNet embedding, ~0.7 works well for clean two-speaker conversations;
///   lower values (0.5) over-split, higher values (0.9+) over-merge.
pub fn diarize(
    samples_16k_mono: &[f32],
    num_speakers: i32,
    threshold: f32,
) -> Result<Vec<Segment>, String> {
    use sherpa_onnx::{
        FastClusteringConfig, OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
        OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
        SpeakerEmbeddingExtractorConfig,
    };

    if !models_ready() {
        return Err(
            "Sherpa-onnx diarization models are not downloaded. Use the Sherpa Models manager first."
                .to_string(),
        );
    }

    let seg_path = segmentation_path()
        .to_str()
        .ok_or_else(|| "Segmentation model path is not valid UTF-8".to_string())?
        .to_string();
    let emb_path = embedding_path()
        .to_str()
        .ok_or_else(|| "Embedding model path is not valid UTF-8".to_string())?
        .to_string();

    let cfg = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(seg_path),
            },
            ..Default::default()
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(emb_path),
            ..Default::default()
        },
        clustering: FastClusteringConfig {
            num_clusters: num_speakers,
            threshold,
        },
        ..Default::default()
    };

    let sd = OfflineSpeakerDiarization::create(&cfg)
        .ok_or_else(|| "Failed to init sherpa diarization".to_string())?;

    if sd.sample_rate() != 16000 {
        return Err(format!(
            "Sherpa expected 16kHz audio, got {}",
            sd.sample_rate()
        ));
    }

    let result = sd
        .process(samples_16k_mono)
        .ok_or_else(|| "Sherpa diarization returned no result".to_string())?;
    let segs = result.sort_by_start_time();

    Ok(segs
        .into_iter()
        .map(|s| Segment {
            start: s.start as f64,
            end: s.end as f64,
            speaker: s.speaker,
        })
        .collect())
}

/// For each Whisper segment, find which Sherpa speaker overlaps most and assign that speaker.
/// Returns Vec<(start, end, text, speaker_letter)>.
pub fn assign_speakers_to_segments(
    whisper_segments: &[(f64, f64, String)],
    sherpa_segments: &[Segment],
) -> Vec<(f64, f64, String, String)> {
    whisper_segments
        .iter()
        .map(|(ws, we, text)| {
            let mut best_speaker: i32 = 0;
            let mut best_overlap: f64 = 0.0;
            for s in sherpa_segments {
                let overlap = (we.min(s.end) - ws.max(s.start)).max(0.0);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_speaker = s.speaker;
                }
            }
            let letter = speaker_letter(best_speaker);
            (*ws, *we, text.clone(), letter)
        })
        .collect()
}

pub fn speaker_letter(n: i32) -> String {
    let n = if n < 0 { 0 } else { n };
    let c = (b'A' + (n as u8 % 26)) as char;
    c.to_string()
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

/// Parse an existing whisper-produced SRT into (start_secs, end_secs, text) tuples.
/// Used by the Sherpa engine to skip Whisper inference when a previous transcription
/// is already on disk for the selected model.
pub fn parse_srt(content: &str) -> Vec<(f64, f64, String)> {
    let mut out = Vec::new();
    for block in content.split("\n\n") {
        let lines: Vec<&str> = block.lines().filter(|l| !l.is_empty()).collect();
        if lines.len() < 2 {
            continue;
        }
        // Time line is the first one containing "-->"
        let time_idx = lines.iter().position(|l| l.contains("-->"));
        let Some(idx) = time_idx else { continue };
        let parts: Vec<&str> = lines[idx].split("-->").collect();
        if parts.len() != 2 {
            continue;
        }
        let start = srt_time_to_secs(parts[0].trim());
        let end = srt_time_to_secs(parts[1].trim());
        let text = lines[idx + 1..].join(" ").trim().to_string();
        if !text.is_empty() {
            out.push((start, end, text));
        }
    }
    out
}

fn srt_time_to_secs(t: &str) -> f64 {
    // "HH:MM:SS,mmm"
    let parts: Vec<&str> = t.split(|c: char| c == ':' || c == ',').collect();
    if parts.len() != 4 {
        return 0.0;
    }
    let h: f64 = parts[0].parse().unwrap_or(0.0);
    let m: f64 = parts[1].parse().unwrap_or(0.0);
    let s: f64 = parts[2].parse().unwrap_or(0.0);
    let ms: f64 = parts[3].parse().unwrap_or(0.0);
    h * 3600.0 + m * 60.0 + s + ms / 1000.0
}

pub fn segments_to_srt(segs: &[(f64, f64, String, String)]) -> String {
    let mut out = String::new();
    for (i, (start, end, text, speaker)) in segs.iter().enumerate() {
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!("{} --> {}\n", fmt_srt_time(*start), fmt_srt_time(*end)));
        out.push_str(&format!("Speaker {}: {}\n\n", speaker, text.trim()));
    }
    out
}

pub fn segments_to_text(segs: &[(f64, f64, String, String)]) -> String {
    let mut out = String::new();
    for (_start, _end, text, speaker) in segs {
        out.push_str(&format!("Speaker {}: {}\n\n", speaker, text.trim()));
    }
    out
}

/// Output JSON in AssemblyAI-style utterance shape so the speaker rename modal works as-is.
pub fn segments_to_aai_json(
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
        "engine": "sherpa-onnx",
        "whisper_model": whisper_model,
        "audio_duration": duration,
        "utterances": utterances,
    })
}
