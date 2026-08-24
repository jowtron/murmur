# Murmur — Project Notes for Claude

Tauri 2 desktop app for bulk audio transcription with optional speaker diarization. Rust backend, TypeScript+Vite frontend (vanilla, no framework). macOS Apple Silicon only.

GitHub: `jowtron/murmur`. Releases: `v0.1.0` → `v0.4.x` with bundled `Whisper.Transcriber.app.zip`.

## Build

Always build the .app bundle from the project root with the deployment-target overrides:

```bash
CMAKE_OSX_DEPLOYMENT_TARGET=10.15 MACOSX_DEPLOYMENT_TARGET=10.15 npx tauri build --bundles app
```

`--bundles app` produces `src-tauri/target/release/bundle/macos/Murmur.app`. Without that flag, Tauri tries to make a `.dmg` and fails.

If the build fails with `failed to read plugin permissions` pointing at the old `audiobooks/whisper-transcriber/` path, stale build-script outputs from the directory rename are to blame: `rm -rf src-tauri/target/{debug,release}/build/tauri-*` and rebuild (keeps the expensive whisper.cpp/sherpa caches).

`CMAKE_OSX_DEPLOYMENT_TARGET=10.15` is required because whisper.cpp's `std::filesystem` needs 10.15+. Setting only `MACOSX_DEPLOYMENT_TARGET` is not enough — the cmake crate overrides it.

For a fast dev iteration: `cd src-tauri && cargo check` (~3s). Frontend typecheck: `npx tsc --noEmit`.

Project lives at `/Users/joseph/Claude_Code/murmur/`. Cargo package, lib, and binary are all `murmur`. Bundle identifier: `com.jowtron.murmur`. Data dir: `~/Library/Application Support/murmur/{models,sherpa-models,templates}` (one-shot startup migration in `lib.rs::migrate_legacy_data_dir` moves a legacy `whisper-transcriber/` dir over).

## Layout

- `index.html` — single-page UI markup. Top settings bar, queue, modals.
- `src/main.ts` — all frontend logic. Vanilla TS, no framework. Long file (~3K lines).
- `src/styles.css` — vanilla CSS.
- `src-tauri/src/lib.rs` — Tauri command registration.
- `src-tauri/src/commands.rs` — all Tauri commands. Long (~1.7K lines).
- `src-tauri/src/transcriber.rs` — Whisper bindings, model enum, audio decode/resample.
- `src-tauri/src/assemblyai.rs` / `deepgram.rs` / `sherpa.rs` — per-engine modules.
- `src-tauri/src/yamnet.rs` — TFLite-based speech/music classifier.
- `src-tauri/src/gap_detection.rs` — energy-based silence detection (legacy fallback).
- `src-tauri/src/mp4_chapters.rs` — reads chapters already embedded in MP4/M4A/M4B containers.
- `src-tauri/src/template.rs` — audio fingerprint template matching (NCC).
- `src-tauri/src/flac_utils.rs` — FLAC seek table repair, chapter embedding via bundled `metaflac`.
- `src-tauri/binaries/` — bundled `metaflac` + `libFLAC` + `libogg` (ad-hoc codesigned, `@executable_path` rpaths).
- `src-tauri/resources/` — bundled `libtensorflowlite_c.dylib` and `yamnet.tflite`.

## Engines (per-file `Engine` dropdown)

| Engine | Module | Cost | Network |
|--------|--------|------|---------|
| Whisper (local) | `transcriber.rs` | Free | Offline |
| Whisper + Sherpa | `sherpa.rs` (+ Whisper) | Free | Offline |
| AssemblyAI | `assemblyai.rs` | $0.17–$0.23/hr | Cloud |
| Deepgram | `deepgram.rs` | ~$0.26/hr | Cloud |

All diarization output uses the same `<basename>.diarized.{srt,txt,json}` naming and AssemblyAI-shape JSON, so the speaker-rename modal works uniformly across engines.

## Conventions

- HTTP calls use **`curl`** subprocess, not `reqwest`. Keeps the dep tree small and matches the pre-existing podcast download pattern.
- Any curl call carrying an API key goes through `curl_util::run_curl` — the Authorization header is piped via `curl -K -` stdin (never on argv, where `ps` exposes it), and an optional `CancellationToken` kills the curl child mid-transfer. Callers pass `--fail-with-body` themselves; the LLM chapter-correction call deliberately omits it to parse error-JSON bodies.
- Engine commands manage cancel tokens via `CancelTokenGuard::register` (drop-guard removes the map entry on every exit path) and temp uploads via `TempFileGuard`. Don't insert/remove `state.cancel_tokens` manually.
- Tauri events are the channel for backend → frontend progress: `transcription-progress`, `model-download-progress`, `podcast-download-progress`, `gap-progress`, `template-match-progress`, `sherpa-model-progress`. Reuse the existing channels rather than inventing new ones.
- Queue items have an `engine` field (`whisper` | `assemblyai` | `deepgram` | `sherpa`); `transcribeItem` dispatches on that.
- Podcast feed downloads are named `YYYY-MM-DD_[E###_]Title.ext` by `feedFilename()` in `main.ts` (pubDate leads so folders sort chronologically; `E###` only when the feed publishes `<itunes:episode>`). `feedLegacyFilename()` reproduces the older title-only, **untrimmed** naming and is checked as a fallback in the existence scan so pre-existing libraries aren't re-downloaded — don't "tidy" it, it has to match bytes already on disk.
- For prompts that need a return value (confirm dialogs), use `await ask(...)` from `@tauri-apps/plugin-dialog` — **not** `window.confirm()`. The native one doesn't reliably block under WKWebView.
- All cancellable jobs share a single `CancellationToken` map keyed by job id (`AppState.cancel_tokens`). New engines should plug into it.
- Concurrency is governed by a single `tokio::sync::Semaphore` shared across all engines.
- CSS class scheme for engine-aware UI dimming:
  - `.whisper-only` — controls Whisper *and* Whisper+Sherpa both use (Model, Threads). Dimmed only under `.engine-cloud`.
  - `.whisper-features` — controls only plain Whisper uses (Format, Per-word, all chapter detection). Dimmed under `.engine-cloud` AND `.engine-sherpa`.

## Gotchas

- **Whisper model download progress** depends on a hardcoded `expected_sizes` HashMap in `download_model`. If you add a new model, also add its byte count there or progress goes wonky (e.g. "940 / 477 MB").
- **HuggingFace LFS downloads can stall** mid-stream (signed-URL expiry, CDN timeouts). curl args include `--retry 5 --retry-all-errors --speed-limit 10000 --speed-time 30 -C -` for resilience. Partial files are kept (not deleted) on failure so the next attempt resumes.
- **Distil model URLs are not on the ggerganov/whisper.cpp repo** — they live on the `distil-whisper` org with quirky filenames (e.g. `ggml-medium-32-2.en.bin` for distil-medium). `WhisperModel::url()` is per-variant.
- **Sherpa-onnx requires 16 kHz mono f32 PCM exactly**. Reuse `transcriber::audio_to_pcm` (Rubato sinc resample) rather than rolling new decode.
- **`tauri build` will fail** with a confusing error if invoked from anywhere other than the project root. Always `cd /Users/joseph/Claude_Code/murmur` first.
- **AssemblyAI submit body** uses the **plural** `speech_models` array for priority routing, not the deprecated `speech_model` singular.
- **Cloud cost rates are hardcoded in two places** — the `CLOUD_RATES` table in `src/main.ts` (drives the per-job "Done" cost badge) and the dropdown labels + About-modal pricing table in `index.html`. A pricing change must update both. All-in rates fold in the always-on diarization add-on: AAI Pro $0.23/hr, AAI U2 $0.17/hr, Deepgram Nova-3 ~$0.26/hr. The badge estimates from the API-billed `duration_secs` (the trimmed length when AAI silence-trim is on); AAI "auto" is priced at the Pro rate so it never understates.
- **Podcast feeds lie about content type.** Supercast served a 1080p H.264 MP4 as `type="audio/mpeg"` with a `.mp3` URL, so neither the enclosure MIME type nor the URL extension is trustworthy. `sniff_extension`/`fix_extension` in `commands.rs` read the first 12 bytes after download and correct the extension (`ftyp` → mp4/m4a/mov by brand, plus `fLaC`/`OggS`/`RIFF`); mp3/aac/opus have no checked magic so they're left alone. This means `download_podcast_episode` can return a **different path than it was given** — callers must use the return value, not the requested path.
- **YAMNet model load** is via `libloading` FFI to `libtensorflowlite_c.dylib`, not native bindings. There used to be an `ort`/ONNX implementation; that was removed.
- **m4b files carry their own chapters, and they beat anything we can detect.** `mp4_chapters.rs` reads them; symphonia's isomp4 demuxer ignores chapters entirely, so it is a hand-rolled box walk. Two schemes exist and both are handled: the **QuickTime chapter track** (the audio trak's `tref/chap` names a text track whose samples are the titles, timed by that track's `stts`) and the Nero **`chpl`** box in `moov/udta`. Audible/Libation `.m4b` files use the chapter track; ffmpeg writes both. The chapter track wins when both are present. Parse failures return an empty list, never an error — a file without chapters is normal.
- **Embedded chapters bypass LLM chapter detection.** `runChapterDetection` in `main.ts` calls `read_embedded_chapters` first; if the file has any, it writes `<stem>_chapters_embedded.{txt,json}` plus the usual .cue/FLAC-embed and returns without touching OpenRouter. The API-key check now sits *after* that probe, so chaptered m4b files need no key at all.
- **The plain-text transcript is chaptered when the source has chapters.** `transcriber::to_chaptered_text` interleaves `=== Title [HH:MM:SS] ===` headings into the `.txt` output for the Whisper and Parakeet jobs. Chapters starting past `duration_secs` are dropped — a trimmed file whose chapter list was copied wholesale would otherwise end in a run of empty headings. The `.srt`/`.vtt`/`.json` outputs and the cloud engines' `.diarized.txt` are untouched.
- **Decode progress is a real stage.** `audio_to_pcm_with_progress` reports `decoding`/`resampling` through `transcriber::StageProgress` (`FnMut(f32, &str)`), because a multi-hour audiobook spends minutes decoding before the ASR emits anything and the UI used to sit at 0% looking hung. Each engine weights the stages itself (Whisper gives decode the first 15%). `audio_to_pcm` is still there as a no-progress wrapper for the callers that don't need it.
- **Testing chapter code with ffmpeg needs `-map_chapters`, not `-map_metadata`.** With only `-map_metadata` the output silently inherits the *source's* full chapter list, times and all, which looks like a parser bug.

## Test files

`/Users/joseph/Claude_Code/audiobooks/disney story time/PT *.flac` — the original use case.

## Workflow

User-driven. Don't push, tag, or release without explicit confirmation. Build commands are safe to run; `git push origin <branch>` and `git tag` are not.

When adding new engines, follow the AssemblyAI/Deepgram pattern: own module, own command, output to `.diarized.{srt,txt,json}` in AssemblyAI-shape JSON, register in `lib.rs`, add to the engine dropdown and `transcribeItem` dispatch in `main.ts`. The speaker-rename modal will work without modification.
