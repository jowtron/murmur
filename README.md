# Murmur

A desktop app for bulk audio transcription and automatic chapter detection, built for audiobook and podcast workflows.

Murmur transcribes audio files locally using OpenAI's Whisper (via Metal GPU acceleration on Mac), then detects chapter boundaries using an LLM and refines them with YAMNet audio classification. Results can be embedded as FLAC metadata, exported as CUE sheets, and manually fine-tuned with a waveform-based alignment tool.

![macOS](https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-blue)

## Features

### Transcription
- **Local Whisper inference** with Metal GPU acceleration (no cloud API needed)
- **Bulk processing** — queue hundreds of files with configurable concurrency
- **Multiple output formats** — SRT, VTT, TXT, JSON (or all at once)
- **Per-word timestamps** option for precise alignment
- **Model manager** — download and switch between Whisper models (tiny through large-v3-turbo)
- **Podcast feed support** — paste an RSS URL to download and queue episodes

### Chapter Detection
- **LLM-powered** — sends transcripts to any OpenRouter-compatible API (Gemini, Claude, GPT, etc.) to identify chapter/story boundaries
- **SRT timestamp correction** — cross-references LLM results against the transcript's own timestamps to fix inaccuracies
- **YAMNet speech onset snapping** — uses Google's YAMNet audio classifier (via TensorFlow Lite) to snap chapter boundaries to the nearest music-to-speech transition
- **Pipeline comparison view** — click the chapter count to see a table comparing where each stage (Raw LLM, SRT Corrected, YAMNet Snap) placed each boundary
- **Waveform alignment modal** — manually drag chapter boundaries on the waveform with audio playback, colored pipeline markers (red = Raw LLM, green = SRT, blue = YAMNet), and visual comparison
- **CUE sheet support** — bulk load and align multiple CUE files
- **Audio template matching** — create fingerprint templates of recurring markers (e.g. intro jingles) and find them across files using cross-correlation

### Output
- **FLAC chapter embedding** — writes chapter markers as FLAC metadata tags
- **CUE sheet export** — generates standard CUE files for CD/player compatibility
- **FLAC seek table repair** — automatically detects FLAC files with missing or inaccurate seek tables and offers to fix them (adds seek points every 1 second using metaflac), ensuring accurate seeking in media players

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **App framework** | [Tauri 2](https://tauri.app/) (Rust backend + web frontend) |
| **Transcription** | [whisper-rs](https://github.com/tazz4843/whisper-rs) (whisper.cpp bindings with Metal) |
| **Audio decoding** | [Symphonia](https://github.com/pdeljanov/Symphonia) (pure Rust, supports FLAC/MP3/WAV/OGG/AAC/M4A/WMA/Opus) |
| **Audio resampling** | [Rubato](https://github.com/HEnquist/rubato) (high-quality sinc interpolation to 16kHz) |
| **Audio classification** | [YAMNet](https://tfhub.dev/google/yamnet/1) via TensorFlow Lite C API ([libloading](https://github.com/nagisa/rust_libloading) FFI) |
| **Chapter detection** | LLM via OpenRouter API (configurable model/prompt) |
| **FLAC tools** | Bundled [metaflac](https://xiph.org/flac/) for seek table repair and chapter embedding |
| **Frontend** | TypeScript + Vite (vanilla, no framework) |
| **Build** | Cargo + npm, bundled as native .app |

### Bundled Native Libraries
- `libtensorflowlite_c.dylib` (3.7MB) — TFLite C runtime for YAMNet inference
- `yamnet.tflite` (3.9MB) — YAMNet audio classification model
- `metaflac` + `libFLAC` + `libogg` — FLAC metadata tools

## Supported Audio Formats

FLAC, MP3, WAV, OGG Vorbis, AAC, M4A, WMA, Opus — any format Symphonia can decode.

Chapter embedding is FLAC-only. CUE export works with any format.

## Installation

### Pre-built (macOS Apple Silicon)

Download `Whisper.Transcriber.app.zip` from the [Releases](https://github.com/jowtron/murmur/releases) page.

> On first launch macOS may block the app. Right-click > Open, or go to System Settings > Privacy & Security > Open Anyway.

### Building from Source

**Requirements:**
- macOS with Apple Silicon (Metal GPU required for whisper-rs)
- Rust toolchain (`rustup`)
- Node.js 18+ and npm
- Tauri CLI (`cargo install tauri-cli`)

```bash
git clone https://github.com/jowtron/murmur.git
cd murmur

npm install

# Build release .app bundle
CMAKE_OSX_DEPLOYMENT_TARGET=10.15 MACOSX_DEPLOYMENT_TARGET=10.15 npx tauri build --bundles app
```

The built app will be at `src-tauri/target/release/bundle/macos/Whisper Transcriber.app`.

> **Note:** The `CMAKE_OSX_DEPLOYMENT_TARGET` env var is required because whisper.cpp uses `std::filesystem` which needs macOS 10.15+. Setting only `MACOSX_DEPLOYMENT_TARGET` is not sufficient — the cmake crate overrides it.

## Usage

### Quick Start

1. Launch the app
2. Click **Models** and download a Whisper model (Large V3 Turbo recommended)
3. Add audio files via **Add Files**, **Add Folder**, or drag-and-drop
4. Click **Transcribe All**

### Chapter Detection

1. Enter your OpenRouter API key in **Settings**
2. Check **Auto-detect chapters** to run chapter detection automatically after transcription
3. Click the chapter count badge (e.g. "6 chapters") on any completed item to see the pipeline comparison table
4. Click **Align** to open the waveform alignment modal and fine-tune boundaries
5. Click **Save Chapters** in the alignment modal to write the final CUE/FLAC

### Settings Reference

Hover over any control for a tooltip description. Key options:

| Setting | Description |
|---------|-------------|
| **Correct times from SRT** | Refine LLM timestamps by matching chapter titles in the transcript |
| **Snap to speech onset** | Use YAMNet to find the nearest music-to-speech transition |
| **First chapter at 0:00** | Force the first chapter to start at the beginning |
| **Embed in FLAC** | Write chapter markers as FLAC metadata |
| **Write .cue** | Generate a CUE sheet alongside the audio |

### FLAC Seek Table Repair

When FLAC files are added to the queue, Murmur checks whether they have proper seek tables. Files without seek tables cause inaccurate seeking in most media players (jumping to a chapter lands at the wrong position). If missing tables are detected, you'll be prompted to fix them automatically.

### Reprocessing

Click the **Reprocess** button on any completed item to re-run chapter detection with different settings. Reprocessing:
- Re-runs the LLM and YAMNet pipeline (clears their caches)
- Does **not** re-run Whisper transcription (SRT is preserved)
- Does **not** overwrite existing CUE files or FLAC chapter embeds (use Align > Save for that)

## Cross-Platform Compatibility

Murmur currently only builds for **macOS on Apple Silicon**. Cross-platform support faces several challenges:

- **whisper-rs** is compiled with the `metal` feature for GPU acceleration. Building for Linux/Windows would require switching to CUDA or CPU-only mode.
- **Bundled native binaries** (`libtensorflowlite_c.dylib`, `metaflac`, `libFLAC`, `libogg`) are macOS ARM64 binaries. Each platform would need its own prebuilt set.
- **TFLite C API** — the prebuilt dylib is from [tphakala/tflite_c](https://github.com/tphakala/tflite_c) which provides releases for multiple platforms, so Linux/Windows TFLite is feasible.
- **metaflac** is widely available on Linux (via package managers) and Windows (FLAC installer). The bundled binary could be replaced with a system dependency.

In principle, a Linux build is achievable with:
1. Removing the `metal` feature from whisper-rs (or adding `cuda`)
2. Replacing the bundled dylibs with Linux `.so` equivalents
3. Replacing `metaflac` with the system-installed version

Windows would additionally require handling `.dll` equivalents and potentially different audio backend configurations.

## License

MIT
