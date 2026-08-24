//! Parakeet (local) transcription engine.
//!
//! NVIDIA Parakeet-tdt-0.6b-v2 run locally via sherpa-onnx (statically linked,
//! same as the diarization engine). Transcription-only — no diarization. Output
//! reuses the Whisper writers (`transcriber::to_srt`/`to_vtt` and the
//! `TranscriptionResult` JSON), so it produces the same
//! `<basename>_transcription_parakeet.{srt,txt,vtt,json}` files.
//!
//! Parakeet is a *non-streaming* (offline) transducer: it decodes the whole
//! input at once, so feeding a 99-minute file in one shot would blow memory. We
//! chunk on silence with Silero VAD, decode each speech segment through the
//! recognizer, and merge per-token timestamps with the chunk's time offset.

use crate::transcriber::{self, Segment, TranscriptionResult};
use std::path::{Path, PathBuf};
use tokio_util::sync::CancellationToken;

/// Directory that holds the extracted Parakeet model directory.
pub fn models_dir() -> PathBuf {
    let dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("murmur")
        .join("parakeet-models");
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// The directory the official tarball extracts to.
pub fn model_dir() -> PathBuf {
    models_dir().join("sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
}

pub fn encoder_path() -> PathBuf {
    model_dir().join("encoder.int8.onnx")
}
pub fn decoder_path() -> PathBuf {
    model_dir().join("decoder.int8.onnx")
}
pub fn joiner_path() -> PathBuf {
    model_dir().join("joiner.int8.onnx")
}
pub fn tokens_path() -> PathBuf {
    model_dir().join("tokens.txt")
}

/// Silero VAD lives in the Whisper models dir (shared single copy).
pub fn silero_vad_path() -> PathBuf {
    transcriber::models_dir().join("silero_vad_v5.onnx")
}

/// All Parakeet ASR files present *and* the Silero VAD model available.
pub fn model_ready() -> bool {
    encoder_path().exists()
        && decoder_path().exists()
        && joiner_path().exists()
        && tokens_path().exists()
        && silero_vad_path().exists()
}

const SAMPLE_RATE: i32 = 16000;

// Segment-grouping heuristics for turning per-token timestamps into readable
// subtitle lines. Parakeet emits BPE subword tokens (word starts marked with
// the '▁' meta symbol); we reassemble words, then group words into lines.
const MAX_WORDS_PER_SEGMENT: usize = 14;
const MAX_SEGMENT_SECS: f64 = 8.0;
const MAX_WORD_GAP_SECS: f64 = 0.8;

/// Transcribe `audio_path` with Parakeet. Decodes to 16 kHz mono PCM, segments
/// on silence with Silero VAD, decodes each speech chunk, and merges results.
///
/// `progress_cb` receives a 0.0–1.0 fraction plus the stage it belongs to
/// (`decoding`, `resampling`, `transcribing`); `cancel` is polled between chunks.
pub fn transcribe(
    audio_path: &Path,
    num_threads: i32,
    cancel: &CancellationToken,
    progress_cb: Option<transcriber::StageProgress>,
) -> Result<TranscriptionResult, String> {
    // Decode + resample to the 16 kHz mono f32 PCM sherpa requires.
    let pcm = transcriber::audio_to_pcm_with_progress(audio_path, progress_cb.clone())?;
    let file_name = audio_path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();
    transcribe_pcm(&pcm, file_name, num_threads, cancel, progress_cb)
}

/// Transcribe pre-decoded 16 kHz mono f32 PCM. Lets a caller that already has
/// the PCM (e.g. the Parakeet+Sherpa engine, which also needs it for
/// diarization) avoid decoding the file twice.
pub fn transcribe_pcm(
    pcm: &[f32],
    file_name: String,
    num_threads: i32,
    cancel: &CancellationToken,
    progress_cb: Option<transcriber::StageProgress>,
) -> Result<TranscriptionResult, String> {
    use sherpa_onnx::{
        OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
        OfflineTransducerModelConfig, SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
    };

    if !model_ready() {
        return Err(
            "Parakeet model is not downloaded. Use the Models manager to download it first."
                .to_string(),
        );
    }

    let to_str = |p: PathBuf, what: &str| {
        p.to_str()
            .map(|s| s.to_string())
            .ok_or_else(|| format!("{} path is not valid UTF-8", what))
    };

    // Build the offline recognizer once and reuse it for every chunk.
    let rec_cfg = OfflineRecognizerConfig {
        model_config: OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: Some(to_str(encoder_path(), "encoder")?),
                decoder: Some(to_str(decoder_path(), "decoder")?),
                joiner: Some(to_str(joiner_path(), "joiner")?),
            },
            tokens: Some(to_str(tokens_path(), "tokens")?),
            num_threads,
            model_type: Some("nemo_transducer".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    let recognizer = OfflineRecognizer::create(&rec_cfg)
        .ok_or_else(|| "Failed to create Parakeet recognizer".to_string())?;

    let duration_secs = pcm.len() as f64 / SAMPLE_RATE as f64;

    if cancel.is_cancelled() {
        return Err("Cancelled".to_string());
    }

    // Silero VAD to split the audio into speech segments. max_speech_duration
    // caps chunk length so a long continuous monologue still fits in memory.
    let vad_cfg = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(to_str(silero_vad_path(), "silero vad")?),
            threshold: 0.5,
            min_silence_duration: 0.5,
            min_speech_duration: 0.25,
            window_size: 512,
            max_speech_duration: 20.0,
        },
        sample_rate: SAMPLE_RATE,
        num_threads,
        ..Default::default()
    };

    // Buffer comfortably larger than max_speech_duration so a long speech run is
    // emitted before the internal circular buffer has to grow (avoids a noisy
    // "Overflow!" log from sherpa — harmless, but the larger buffer sidesteps it).
    let vad = VoiceActivityDetector::create(&vad_cfg, 60.0)
        .ok_or_else(|| "Failed to create Silero VAD".to_string())?;

    // Feed the waveform in window_size frames, draining detected segments as we
    // go. Each segment carries its absolute start sample index and a copy of its
    // samples (kept small — total speech ≤ total audio, already in memory).
    let window = 512usize;
    let mut chunks: Vec<(f64, Vec<f32>, f64)> = Vec::new(); // (offset_secs, samples, end_secs)
    let drain = |vad: &VoiceActivityDetector, out: &mut Vec<(f64, Vec<f32>, f64)>| {
        while !vad.is_empty() {
            if let Some(seg) = vad.front() {
                let offset = seg.start() as f64 / SAMPLE_RATE as f64;
                let samples = seg.samples().to_vec();
                let end = offset + samples.len() as f64 / SAMPLE_RATE as f64;
                out.push((offset, samples, end));
            }
            vad.pop();
        }
    };

    let mut i = 0usize;
    while i < pcm.len() {
        if cancel.is_cancelled() {
            return Err("Cancelled".to_string());
        }
        let end = (i + window).min(pcm.len());
        vad.accept_waveform(&pcm[i..end]);
        i = end;
        drain(&vad, &mut chunks);
    }
    vad.flush();
    drain(&vad, &mut chunks);

    // Decode each speech chunk and merge into a single segment list, offsetting
    // every timestamp by the chunk's position in the original audio.
    let mut segments: Vec<Segment> = Vec::new();
    let total = chunks.len().max(1);
    for (idx, (offset, samples, chunk_end)) in chunks.iter().enumerate() {
        if cancel.is_cancelled() {
            return Err("Cancelled".to_string());
        }

        let stream = recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE, samples);
        recognizer.decode(&stream);

        if let Some(result) = stream.get_result() {
            let text = result.text.trim().to_string();
            if !text.is_empty() {
                let mut chunk_segs = build_segments(
                    *offset,
                    *chunk_end,
                    &result.tokens,
                    result.timestamps.as_deref(),
                    result.durations.as_deref(),
                    &text,
                );
                segments.append(&mut chunk_segs);
            }
        }

        if let Some(cb) = &progress_cb {
            if let Ok(mut f) = cb.lock() {
                f((idx + 1) as f32 / total as f32, "transcribing");
            }
        }
    }

    let full_text = segments
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();

    Ok(TranscriptionResult {
        file: file_name,
        segments,
        text: full_text,
        duration_secs,
    })
}

/// Turn one chunk's tokens + per-token timestamps into subtitle-sized segments.
///
/// Parakeet/NeMo BPE tokens mark word starts with the '▁' (U+2581) meta symbol.
/// We reassemble words, then group them into lines bounded by word count,
/// duration, and inter-word gaps. If timestamps are missing or inconsistent we
/// fall back to a single segment spanning the whole chunk.
fn build_segments(
    offset: f64,
    chunk_end: f64,
    tokens: &[String],
    timestamps: Option<&[f32]>,
    durations: Option<&[f32]>,
    full_text: &str,
) -> Vec<Segment> {
    let Some(ts) = timestamps else {
        return vec![Segment {
            start: offset,
            end: chunk_end,
            text: full_text.to_string(),
        }];
    };
    if ts.len() != tokens.len() || tokens.is_empty() {
        return vec![Segment {
            start: offset,
            end: chunk_end,
            text: full_text.to_string(),
        }];
    }

    // Reassemble subword tokens into words with start/end times.
    struct Word {
        start: f64,
        end: f64,
        text: String,
    }
    let mut words: Vec<Word> = Vec::new();
    for (k, tok) in tokens.iter().enumerate() {
        let start = offset + ts[k] as f64;
        let end = durations
            .and_then(|d| d.get(k))
            .map(|d| start + *d as f64)
            .unwrap_or(start);
        let is_word_start = tok.starts_with('\u{2581}');
        let piece = tok.trim_start_matches('\u{2581}');
        if is_word_start || words.is_empty() {
            words.push(Word {
                start,
                end,
                text: piece.to_string(),
            });
        } else if let Some(last) = words.last_mut() {
            last.text.push_str(piece);
            last.end = end.max(last.end);
        }
    }

    // Word end times: prefer the explicit duration; otherwise extend to the next
    // word's start so lines don't show zero-length spans.
    for k in 0..words.len() {
        if words[k].end <= words[k].start {
            words[k].end = if k + 1 < words.len() {
                words[k + 1].start
            } else {
                chunk_end
            };
        }
    }

    // Group words into lines.
    let mut segs: Vec<Segment> = Vec::new();
    let mut cur: Vec<&Word> = Vec::new();
    let flush = |cur: &mut Vec<&Word>, segs: &mut Vec<Segment>| {
        if cur.is_empty() {
            return;
        }
        let start = cur.first().unwrap().start;
        let end = cur.last().unwrap().end;
        let text = cur
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
            .trim()
            .to_string();
        if !text.is_empty() {
            segs.push(Segment { start, end, text });
        }
        cur.clear();
    };

    for w in &words {
        if let Some(prev) = cur.last() {
            let gap = w.start - prev.end;
            let dur = w.end - cur.first().unwrap().start;
            if cur.len() >= MAX_WORDS_PER_SEGMENT
                || dur > MAX_SEGMENT_SECS
                || gap > MAX_WORD_GAP_SECS
            {
                flush(&mut cur, &mut segs);
            }
        }
        cur.push(w);
    }
    flush(&mut cur, &mut segs);

    if segs.is_empty() {
        segs.push(Segment {
            start: offset,
            end: chunk_end,
            text: full_text.to_string(),
        });
    }
    segs
}
