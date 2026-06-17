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
    // WeSpeaker CAM++ (English VoxCeleb, large-margin finetuned) — current
    // state-of-the-art open speaker embedding. Replaced the older, weaker
    // NeMo SpeakerNet. CAM++ lives in a different embedding space, so the
    // clustering threshold default was re-tuned (~0.5, see `diarize`).
    models_dir().join("wespeaker_en_voxceleb_CAM++_LM.onnx")
}

/// Legacy SpeakerNet embedding path — kept only so a one-time cleanup can
/// remove the orphaned 23 MB file after the upgrade to CAM++.
pub fn legacy_embedding_path() -> PathBuf {
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
/// `threshold` — distance threshold for the FastClustering algorithm. With the
///   WeSpeaker CAM++ embedding, ~0.5 works well for clean conversations; lower
///   values over-split, higher values over-merge. (Ignored when num_speakers > 0.)
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

/// Merge consecutive segments that share a speaker into a single turn, so output
/// reads as one paragraph per speaker turn instead of one line per tiny Whisper
/// segment. Mirrors how AssemblyAI groups consecutive same-speaker words into an
/// utterance — without this, word-level Whisper segments produce a wall of
/// one-line "Speaker: 3 words" fragments. Applies to every diarization engine.
pub fn coalesce_by_speaker(
    segs: &[(f64, f64, String, String)],
) -> Vec<(f64, f64, String, String)> {
    let mut out: Vec<(f64, f64, String, String)> = Vec::new();
    for (start, end, text, speaker) in segs {
        let t = text.trim();
        match out.last_mut() {
            Some(last) if &last.3 == speaker => {
                last.1 = *end;
                if !t.is_empty() {
                    if !last.2.is_empty() {
                        last.2.push(' ');
                    }
                    last.2.push_str(t);
                }
            }
            _ => out.push((*start, *end, t.to_string(), speaker.clone())),
        }
    }
    out
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

/// Human-readable identity of the speaker-embedding model currently in use,
/// derived from the on-disk filename so it can never drift from reality.
pub fn embedding_model_id() -> String {
    embedding_path()
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Output JSON in AssemblyAI-style utterance shape so the speaker rename modal works as-is.
pub fn segments_to_aai_json(
    segs: &[(f64, f64, String, String)],
    duration: f64,
    whisper_model: &str,
    num_speakers: i32,
    threshold: f32,
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
        // Diarization provenance — so an output file self-documents exactly how
        // it was produced and you can tell which embedding/settings were used.
        "embedding_model": embedding_model_id(),
        "diarization_num_speakers": if num_speakers > 0 { serde_json::json!(num_speakers) } else { serde_json::json!("auto") },
        "diarization_threshold": threshold,
        "audio_duration": duration,
        "utterances": utterances,
    })
}
