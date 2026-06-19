import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { open, ask } from "@tauri-apps/plugin-dialog";

// Injected by Vite at build time (vite.config.ts): "<git-hash>[-dirty] · MM-DD HH:MM".
declare const __BUILD_ID__: string;
{
  const el = document.getElementById("build-info");
  if (el) el.textContent = __BUILD_ID__;
}

interface ModelInfo {
  name: string;
  display_name: string;
  downloaded: boolean;
  filename: string;
  size_bytes: number;
}

// Shape returned by transcribe_file / transcribe_parakeet (Rust TranscriptionResult).
interface TranscriptionResult {
  file: string;
  segments: { start: number; end: number; text: string }[];
  text: string;
  duration_secs: number;
}

interface TranscriptionProgress {
  job_id: string;
  file: string;
  progress: number;
  status: string;
}

interface ModelDownloadProgress {
  model: string;
  progress: number;
  status: string;
  downloaded_mb?: number;
  total_mb?: number;
}

interface GpuInfo {
  name: string;
  gpu_cores: number | null;
  metal_supported: boolean;
  using_metal: boolean;
}

interface Chapter {
  title: string;
  start_time: string;
  start_secs: number;
}

interface SeekInfo {
  seekable: boolean;
  has_seektable: boolean;
  fixable: boolean;
  format: string;
  message: string | null;
}

interface ChapterWithSnap {
  title: string;
  start_time: string;
  start_secs: number;
  raw_llm_secs: number;
  srt_corrected_secs: number | null;
  yamnet_secs: number | null;
  snapped: boolean;
}

type Engine = "whisper" | "assemblyai" | "deepgram" | "sherpa" | "parakeet" | "parakeet-sherpa" | "compare-local";

interface QueueItem {
  id: string;
  path: string;
  name: string;
  duration: number | null;
  status: "pending" | "queued" | "transcribing" | "detecting" | "complete" | "error" | "cancelled";
  progress: number;
  error?: string;
  modelUsed?: string;
  startedAt?: number;
  elapsed?: number;
  autoDetectChapters: boolean;
  chapters?: Chapter[];
  snappedChapters?: ChapterWithSnap[];
  detectStatus?: string;
  engine: Engine;
  speakerCount?: number;
  stageText?: string;
  diarizedJsonPath?: string;
  diarizedSrtPath?: string;
  diarizedTxtPath?: string;
  speakerNames?: Record<string, string>;
}

const queue: QueueItem[] = [];
let customOutputDir: string | null = null;

// Settings
function loadSettings() {
  return {
    apiKey: localStorage.getItem("openrouter_api_key") || (document.getElementById("input-api-key") as HTMLInputElement)?.value || "",
    llmModel: selectLlmModel.value || localStorage.getItem("llm_model") || "google/gemini-3.1-flash-lite-preview",
    llmModels: localStorage.getItem("llm_models") || "google/gemini-3.1-flash-lite-preview\ngoogle/gemini-2.5-flash-preview\nanthropic/claude-sonnet-4\nopenai/gpt-4o-mini",
    apiUrl: localStorage.getItem("api_url") || "https://openrouter.ai/api/v1/chat/completions",
    chapterPrompt: localStorage.getItem("chapter_prompt") || (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement)?.value || "",
    chapterOutputFormat: localStorage.getItem("chapter_output_format") || "json",
    assemblyaiKey: localStorage.getItem("assemblyai_api_key") || (document.getElementById("input-assemblyai-key") as HTMLInputElement)?.value || "",
    assemblyaiModel: localStorage.getItem("assemblyai_model") || "auto",
    assemblyaiLang: localStorage.getItem("assemblyai_lang") ?? "en",
    assemblyaiTrimSilence: localStorage.getItem("assemblyai_trim_silence") === "1",
    assemblyaiSilenceDb: parseFloat(localStorage.getItem("assemblyai_silence_db") || "-35"),
    assemblyaiMinSilenceSecs: parseFloat(localStorage.getItem("assemblyai_min_silence_secs") || "0.75"),
    assemblyaiSilencePadSecs: parseFloat(localStorage.getItem("assemblyai_silence_pad_secs") || "0.1"),
    deepgramKey: localStorage.getItem("deepgram_api_key") || (document.getElementById("input-deepgram-key") as HTMLInputElement)?.value || "",
    deepgramModel: localStorage.getItem("deepgram_model") || "nova-3",
    deepgramLang: localStorage.getItem("deepgram_lang") ?? "en",
    // Shared "Speakers expected" hint (top settings bar). Blank/0 = auto.
    // Feeds AssemblyAI's speakers_expected and Sherpa's num_speakers.
    speakersExpected: speakersExpectedValue(),
    sherpaNumSpeakers: speakersExpectedValue() ?? 0,
    sherpaThreshold: parseFloat(localStorage.getItem("sherpa_threshold") || "0.5"),
  };
}

/// Read the shared "Speakers" field. Returns a clamped 1..=10 integer, or null
/// for blank/invalid (auto-detect). Reads the LIVE input element first so a
/// value typed but not yet committed (no blur/change event) is still honoured
/// at Transcribe time; falls back to saved prefs only if the element is absent.
function speakersExpectedValue(): number | null {
  const el = document.getElementById("input-speakers-expected") as HTMLInputElement | null;
  const raw = (el?.value?.trim())
    || localStorage.getItem("speakers_expected")
    || localStorage.getItem("sherpa_num_speakers")
    || "";
  const n = parseInt(raw, 10);
  if (!Number.isFinite(n) || n < 1) return null;
  return Math.min(n, 10);
}

function assemblyaiSpeechModels(choice: string): string[] {
  switch (choice) {
    case "universal-3-pro": return ["universal-3-pro"];
    case "universal-2": return ["universal-2"];
    case "auto":
    default: return ["universal-3-pro", "universal-2"];
  }
}

function llmModelShort(fullModel: string): string {
  // "google/gemini-3.1-flash-lite-preview" → "gemini-3.1-flash-lite-preview"
  return fullModel.split("/").pop() || fullModel;
}

function refreshLlmDropdown() {
  const models = (localStorage.getItem("llm_models") || "google/gemini-3.1-flash-lite-preview").split("\n").map(m => m.trim()).filter(Boolean);
  const current = selectLlmModel.value || localStorage.getItem("llm_model") || models[0];
  selectLlmModel.innerHTML = models.map(m =>
    `<option value="${escapeHtml(m)}" ${m === current ? "selected" : ""}>${escapeHtml(llmModelShort(m))}</option>`
  ).join("");
}

function saveSettings() {
  localStorage.setItem("openrouter_api_key", (document.getElementById("input-api-key") as HTMLInputElement).value);
  localStorage.setItem("assemblyai_api_key", (document.getElementById("input-assemblyai-key") as HTMLInputElement).value);
  localStorage.setItem("assemblyai_model", (document.getElementById("select-assemblyai-model") as HTMLSelectElement).value);
  localStorage.setItem("assemblyai_lang", (document.getElementById("input-assemblyai-lang") as HTMLInputElement).value);
  localStorage.setItem("assemblyai_trim_silence", (document.getElementById("chk-aai-trim-silence") as HTMLInputElement).checked ? "1" : "0");
  localStorage.setItem("assemblyai_silence_db", (document.getElementById("input-aai-silence-db") as HTMLInputElement).value);
  localStorage.setItem("assemblyai_min_silence_secs", (document.getElementById("input-aai-min-silence") as HTMLInputElement).value);
  localStorage.setItem("assemblyai_silence_pad_secs", (document.getElementById("input-aai-silence-pad") as HTMLInputElement).value);
  localStorage.setItem("deepgram_api_key", (document.getElementById("input-deepgram-key") as HTMLInputElement).value);
  localStorage.setItem("deepgram_model", (document.getElementById("select-deepgram-model") as HTMLSelectElement).value);
  localStorage.setItem("deepgram_lang", (document.getElementById("input-deepgram-lang") as HTMLInputElement).value);
  localStorage.setItem("sherpa_threshold", (document.getElementById("input-sherpa-threshold") as HTMLInputElement).value);
  const modelsText = (document.getElementById("input-llm-models") as HTMLTextAreaElement).value;
  localStorage.setItem("llm_models", modelsText);
  const modelsList = modelsText.split("\n").map(m => m.trim()).filter(Boolean);
  if (modelsList.length > 0 && !modelsList.includes(selectLlmModel.value)) {
    localStorage.setItem("llm_model", modelsList[0]);
  }
  refreshLlmDropdown();
  localStorage.setItem("api_url", (document.getElementById("input-api-url") as HTMLInputElement).value);
  localStorage.setItem("chapter_prompt", (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement).value);
  localStorage.setItem("chapter_output_format", (document.getElementById("select-chapter-format") as HTMLSelectElement).value);
  localStorage.setItem("align_before", (document.getElementById("input-align-before") as HTMLInputElement).value);
  localStorage.setItem("align_after", (document.getElementById("input-align-after") as HTMLInputElement).value);
}

function getAlignBefore(): number { return parseFloat(localStorage.getItem("align_before") || "2"); }
function getAlignAfter(): number { return parseFloat(localStorage.getItem("align_after") || "2"); }

// DOM
const queueList = document.getElementById("queue-list")!;
const queueCount = document.getElementById("queue-count")!;
const btnAddFiles = document.getElementById("btn-add-files")!;
const btnAddFolder = document.getElementById("btn-add-folder")!;
const btnAddFeed = document.getElementById("btn-add-feed")!;
const btnModels = document.getElementById("btn-models")!;
const btnSettings = document.getElementById("btn-settings")!;
const btnTranscribeAll = document.getElementById("btn-transcribe-all")! as HTMLButtonElement;
const btnClearQueue = document.getElementById("btn-clear-queue")!;
const btnRetryErrors = document.getElementById("btn-retry-errors")! as HTMLButtonElement;
const btnClearDone = document.getElementById("btn-clear-done")! as HTMLButtonElement;
const modelModal = document.getElementById("model-modal")!;
const modelList = document.getElementById("model-list")!;
const modelsDir = document.getElementById("models-dir")!;
const settingsModal = document.getElementById("settings-modal")!;
const feedModal = document.getElementById("feed-modal")!;
const selectEngine = document.getElementById("select-engine")! as HTMLSelectElement;
const selectModel = document.getElementById("select-model")! as HTMLSelectElement;
const selectFormat = document.getElementById("select-format")! as HTMLSelectElement;
const selectOutput = document.getElementById("select-output")! as HTMLSelectElement;
const selectThreads = document.getElementById("select-threads")! as HTMLSelectElement;
const selectConcurrent = document.getElementById("select-concurrent")! as HTMLSelectElement;
const inputSpeakersExpected = document.getElementById("input-speakers-expected")! as HTMLInputElement;
const chkAutoChapters = document.getElementById("chk-auto-chapters")! as HTMLInputElement;
const chkForceOverwrite = document.getElementById("chk-force-overwrite")! as HTMLInputElement;

async function checkOutputExists<T>(cmd: string, args: T): Promise<boolean> {
  if (chkForceOverwrite.checked) return false;
  return await invoke<boolean>(cmd, args as any);
}
const chkSnapGaps = document.getElementById("chk-snap-gaps")! as HTMLInputElement;
const chkSrtCorrect = document.getElementById("chk-srt-correct")! as HTMLInputElement;
const chkFirstZero = document.getElementById("chk-first-zero")! as HTMLInputElement;
const chkPerWord = document.getElementById("chk-per-word")! as HTMLInputElement;
const selectLlmModel = document.getElementById("select-llm-model")! as HTMLSelectElement;
const chkEmbedFlac = document.getElementById("chk-embed-flac")! as HTMLInputElement;
const chkWriteCue = document.getElementById("chk-write-cue")! as HTMLInputElement;
const gpuBar = document.getElementById("gpu-bar")!;

function generateId(): string {
  return crypto.randomUUID();
}

function formatDuration(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  if (h > 0) return `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function formatElapsed(ms: number): string {
  const secs = Math.floor(ms / 1000);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function engineShort(e: Engine): string {
  switch (e) {
    case "assemblyai": return "AssemblyAI";
    case "deepgram": return "Deepgram";
    case "sherpa": return "Sherpa";
    case "parakeet": return "Parakeet";
    case "parakeet-sherpa": return "Parakeet+Sherpa";
    case "compare-local": return "Compare";
    case "whisper":
    default: return "Whisper";
  }
}
function engineLabel(e: Engine): string {
  switch (e) {
    case "assemblyai": return "AssemblyAI cloud diarization";
    case "deepgram": return "Deepgram cloud diarization";
    case "sherpa": return "Whisper + Sherpa local diarization";
    case "parakeet": return "Parakeet local transcription";
    case "parakeet-sherpa": return "Parakeet + Sherpa local diarization";
    case "compare-local": return "Compare all downloaded local transcription models";
    case "whisper":
    default: return "Whisper local transcription";
  }
}

function escapeHtml(text: string): string {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function updateQueueButtons() {
  const hasErrors = queue.some((q) => q.status === "error");
  const hasDone = queue.some((q) => q.status === "complete");
  btnRetryErrors.style.display = hasErrors ? "" : "none";
  btnClearDone.style.display = hasDone ? "" : "none";
}

function renderQueue() {
  if (renderPaused) return;
  queueCount.textContent = queue.length.toString();
  updateQueueButtons();

  if (queue.length === 0) {
    queueList.innerHTML = `<div class="empty-state">Drop audio files here or click "Add Files" / "Add Folder"</div>`;
    return;
  }

  queueList.innerHTML = queue
    .map(
      (item) => `
    <div class="queue-item" data-id="${item.id}">
      <div class="file-info">
        <div class="file-name">
          ${escapeHtml(item.name)}
          <span class="engine-badge engine-${item.engine}" title="${engineLabel(item.engine)}">${engineShort(item.engine)}</span>
          ${item.speakerCount ? `<span class="speaker-badge" title="Distinct speakers detected">${item.speakerCount} speakers</span>` : ""}
        </div>
        <div class="file-path">${escapeHtml(item.path)}</div>
        ${item.error ? `<div class="error-msg">${escapeHtml(item.error)}</div>` : ""}
        ${item.chapters ? `<div class="chapters-badge clickable" data-id="${item.id}" style="cursor:pointer; text-decoration:underline;">${item.chapters.length} chapters detected</div> <span class="chapters-badge" style="cursor:pointer;text-decoration:underline;margin-left:6px;" data-align-id="${item.id}">Align</span>` : ""}
        ${
          item.status === "transcribing" || item.status === "detecting"
            ? `<div class="progress-bar"><div class="fill" style="width: ${Math.round(item.progress * 100)}%"></div></div>`
            : ""
        }
        ${
          item.status === "complete"
            ? `<div class="progress-bar"><div class="fill complete" style="width: 100%"></div></div>`
            : ""
        }
      </div>
      <div class="duration">${item.duration ? formatDuration(item.duration) : "..."}</div>
      <div class="model-col">${item.modelUsed || "-"}</div>
      <div class="elapsed-col">${item.elapsed ? formatElapsed(item.elapsed) : "-"}</div>
      <div class="status-col">
        <span class="status status-${item.status}${item.stageText && item.stageText.endsWith("…") ? " is-working" : ""}">
          ${item.status === "pending" ? "Pending" : ""}
          ${item.status === "queued" ? (item.stageText || "Queued") : ""}
          ${item.status === "transcribing" ? (item.stageText && item.engine !== "whisper" ? (item.progress > 0 && item.progress < 1 ? `${item.stageText} ${Math.round(item.progress * 100)}%` : item.stageText) : `${Math.round(item.progress * 100)}%`) : ""}
          ${item.status === "detecting" ? `<span id="detect-status-${item.id}">Detecting...</span>` : ""}
          ${item.status === "complete" ? "Done" : ""}
          ${item.status === "error" ? "Error" : ""}
          ${item.status === "cancelled" ? "Cancelled" : ""}
        </span>
      </div>
      <div class="actions">
        ${item.status === "transcribing" || item.status === "queued" || item.status === "detecting" ? `<button class="small danger btn-cancel" data-id="${item.id}">Cancel</button>` : ""}
        ${item.status === "error" || item.status === "cancelled" ? `<button class="small btn-retry" data-id="${item.id}">Retry</button>` : ""}
        ${item.status === "complete" && (item.engine === "assemblyai" || item.engine === "deepgram" || item.engine === "sherpa" || item.engine === "parakeet-sherpa") && item.diarizedJsonPath ? `<button class="small btn-speakers" data-id="${item.id}">Identify speakers</button>` : ""}
        ${item.status === "complete" && (item.engine === "whisper" || item.engine === "parakeet") ? `<button class="small btn-retranscribe" data-id="${item.id}" title="Re-run transcription with the model currently selected in the dropdown">Re-transcribe</button>` : ""}
        ${item.status === "complete" && item.engine === "whisper" ? `<button class="small btn-redo-chapters" data-id="${item.id}" title="Re-run LLM chapter detection (needs an OpenRouter API key in Settings)">Redo chapters</button>` : ""}
        ${item.status !== "transcribing" && item.status !== "detecting" ? `<button class="small danger btn-remove" data-id="${item.id}">&times;</button>` : ""}
      </div>
    </div>
  `
    )
    .join("");

  // Attach handlers
  queueList.querySelectorAll(".btn-remove").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = (btn as HTMLElement).dataset.id!;
      const idx = queue.findIndex((q) => q.id === id);
      if (idx !== -1) {
        // Cancel any backend work before removing
        await invoke("cancel_job", { jobId: id }).catch(() => {});
        queue[idx].status = "cancelled";
        queue.splice(idx, 1);
        renderQueue();
      }
    });
  });

  queueList.querySelectorAll(".btn-retry").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = (btn as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (item) {
        item.status = "pending";
        item.progress = 0;
        item.error = undefined;
        item.elapsed = undefined;
        renderQueue();
        transcribeItem(item);
      }
    });
  });

  queueList.querySelectorAll(".btn-cancel").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = (btn as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (item) {
        await invoke("cancel_job", { jobId: id });
        item.status = "cancelled";
        item.error = "Cancelled by user";
        renderQueue();
      }
    });
  });

  queueList.querySelectorAll(".btn-speakers").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = (btn as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (item) openSpeakerModal(item);
    });
  });

  // Clickable chapters badge
  queueList.querySelectorAll(".chapters-badge.clickable").forEach((badge) => {
    badge.addEventListener("click", () => {
      const id = (badge as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (item?.chapters) showChapterDetail(item);
    });
  });

  // Re-transcribe button — re-run transcription with the currently selected
  // model (and current settings). Does NOT touch chapter detection; that's the
  // "Redo chapters" button. Useful for comparing models on the same file.
  queueList.querySelectorAll(".btn-retranscribe").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = (btn as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (!item) return;

      // Re-stamp the engine to whatever the dropdown says now, so switching the
      // engine (e.g. Whisper → Parakeet) and re-transcribing works too.
      item.engine = (selectEngine.value as Engine) || item.engine;
      item.chapters = undefined;
      item.snappedChapters = undefined;
      item.modelUsed = undefined;
      item.error = undefined;
      // Transcription only — leave chapters to the dedicated button.
      item.autoDetectChapters = false;
      item.status = "pending";
      renderQueue();
      await transcribeItem(item);
    });
  });

  // Redo chapters button — re-run LLM chapter detection with current settings.
  queueList.querySelectorAll(".btn-redo-chapters").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const id = (btn as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (!item) return;

      // Delete cached chapter files so they regenerate
      const outputDir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      const stem = item.name.replace(/\.[^.]+$/, "");
      const settings = loadSettings();
      const llmShort = llmModelShort(settings.llmModel);
      const cachePaths = [
        `${outputDir}/${stem}_${llmShort}_llm_chapters.json`,
        `${outputDir}/${stem}_${llmShort}_llm_raw.json`,
        `${outputDir}/${stem}_chapters_${llmShort}.json`,
        `${outputDir}/${stem}_chapters_${llmShort}.txt`,
        `${outputDir}/${stem}_yamnet.json`,
        `${outputDir}/${stem}_yamnet_onsets.json`,
      ];
      for (const p of cachePaths) {
        try { await invoke("delete_file", { path: p }); } catch { /* ignore */ }
      }

      item.chapters = undefined;
      item.snappedChapters = undefined;
      item.status = "detecting";
      item.error = undefined;
      item.autoDetectChapters = true;
      renderQueue();
      await runChapterDetection(item, true);
      if ((item.status as string) !== "cancelled") {
        item.status = "complete";
      }
      renderQueue();
    });
  });

  // Align button
  queueList.querySelectorAll("[data-align-id]").forEach((el) => {
    el.addEventListener("click", (e) => {
      e.stopPropagation();
      const id = (el as HTMLElement).dataset.alignId!;
      const item = queue.find((q) => q.id === id);
      if (item?.chapters) openAlignModal(item.path, item.chapters, item.duration || 0, item.snappedChapters);
    });
  });

  // Auto-scroll to the currently active item
  const activeItem = queue.find(q => q.status === "transcribing" || q.status === "detecting");
  if (activeItem) {
    const el = queueList.querySelector(`[data-id="${activeItem.id}"]`);
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}

function formatHMS(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function showChapterDetail(item: QueueItem) {
  const modal = document.getElementById("chapter-detail-modal")!;
  const title = document.getElementById("chapter-detail-title")!;
  const list = document.getElementById("chapter-detail-list")!;

  title.textContent = `Chapters — ${item.name}`;

  const hasSnap = item.snappedChapters && item.snappedChapters.length > 0;

  let html = "";
  if (hasSnap) {
    // Show comparison table with all pipeline stages
    html += `<table class="chapter-compare-table">
      <thead><tr>
        <th>#</th><th>Title</th><th>Raw LLM</th><th>SRT Corrected</th><th>YAMNet Snap</th><th>Final</th>
      </tr></thead><tbody>`;
    (item.snappedChapters || []).forEach((snap, idx) => {
      let rawSecs = snap.raw_llm_secs;
      if (chkFirstZero.checked && idx === 0) rawSecs = 0;
      const rawTime = formatHMS(rawSecs);
      const srtTime = snap.srt_corrected_secs != null ? formatHMS(snap.srt_corrected_secs) : "—";
      const yamTime = snap.yamnet_secs != null ? formatHMS(snap.yamnet_secs) : "—";
      const finalTime = snap.start_time;

      // Highlight cells that differ from the final time
      const rawClass = rawTime !== finalTime ? "ch-diff" : "ch-match";
      const srtClass = srtTime !== "—" && srtTime !== finalTime ? "ch-diff" : srtTime === finalTime ? "ch-match" : "";
      const yamClass = yamTime !== "—" && yamTime !== finalTime ? "ch-diff" : yamTime === finalTime ? "ch-match" : "";

      html += `<tr>
        <td>${idx + 1}</td>
        <td class="ch-title-cell">${escapeHtml(snap.title)}</td>
        <td class="${rawClass}">${rawTime}</td>
        <td class="${srtClass}">${srtTime}</td>
        <td class="${yamClass}">${yamTime}</td>
        <td class="ch-final">${finalTime}</td>
      </tr>`;
    });
    html += `</tbody></table>`;
  } else {
    // Simple list when no snap data
    html = (item.chapters || []).map((ch) => `
      <div class="chapter-item">
        <span class="ch-time">${escapeHtml(ch.start_time)}</span>
        <span class="ch-title">${escapeHtml(ch.title)}</span>
      </div>`).join("");
  }

  list.innerHTML = html;

  modal.classList.remove("hidden");
  modal.querySelector(".modal-close")!.addEventListener("click", () => modal.classList.add("hidden"), { once: true });
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); }, { once: true });
}

// Elapsed time updater
let renderPaused = false;
setInterval(() => {
  if (renderPaused) return;
  let needsRender = false;
  for (const item of queue) {
    if (item.status === "transcribing" && item.startedAt) {
      item.elapsed = Date.now() - item.startedAt;
      needsRender = true;
    }
  }
  if (needsRender) renderQueue();
}, 1000);

async function addFiles(paths: string[]) {
  const autoChapters = chkAutoChapters.checked;
  const engine = (selectEngine.value as Engine) || "whisper";

  // Check for FLAC files missing seek tables
  const fixablePaths: string[] = [];
  for (const path of paths) {
    try {
      const info = await invoke<SeekInfo>("check_seekability", { path });
      if (info.fixable && !info.has_seektable) {
        fixablePaths.push(path);
      }
    } catch { /* ignore non-FLAC or errors */ }
  }

  if (fixablePaths.length > 0) {
    const doFix = confirm(
      `${fixablePaths.length} FLAC file(s) are missing seek tables.\n` +
      `This causes inaccurate seeking in media players.\n\n` +
      `Fix them now? (adds seek points every 1 second, requires metaflac)`
    );
    if (doFix) {
      try {
        const fixed = await invoke<string[]>("fix_seektables_batch", { paths: fixablePaths });
        if (fixed.length > 0) {
          console.log(`Fixed seek tables in ${fixed.length} files`);
        }
      } catch (err) {
        console.error("Failed to fix seek tables:", err);
      }
    }
  }

  let skipped = 0;
  for (const path of paths) {
    // Dedupe against the queue only (same path + engine). Outputs on disk
    // are deliberately not checked here — the Force overwrite checkbox
    // governs whether existing outputs are redone at processing time.
    if (queue.some((q) => q.path === path && q.engine === engine)) {
      skipped++;
      continue;
    }
    const name = path.split("/").pop() || path;
    const item: QueueItem = {
      id: generateId(),
      path,
      name,
      duration: null,
      status: "pending",
      progress: 0,
      autoDetectChapters: autoChapters && engine === "whisper",
      engine,
    };
    queue.push(item);

    invoke<number>("get_audio_duration", { path })
      .then((dur) => { item.duration = dur; renderQueue(); })
      .catch(() => {});
  }
  renderQueue();
  if (skipped > 0 && skipped === paths.length) {
    alert(`All ${skipped} file(s) are already in the queue.`);
  }
}

async function checkModelAndPromptDownload(modelName: string): Promise<boolean> {
  const ready = await invoke<boolean>("is_model_ready", { name: modelName });
  if (ready) return true;

  const doDownload = await ask(
    `The model "${modelName}" is not downloaded yet.\n\nWould you like to download it now?`,
    { title: "Download model?", kind: "info" }
  );
  if (!doDownload) return false;

  // Open model manager so the user sees the progress bar for this model
  modelModal.classList.remove("hidden");
  await refreshModelList();

  // Trigger the Download button for this specific model so the user
  // doesn't have to find and click it themselves.
  const dlBtn = modelList.querySelector<HTMLButtonElement>(
    `.btn-download-model[data-name="${modelName}"]`
  );
  if (dlBtn) {
    dlBtn.click();
  }

  // Wait for the model to exist or the user to close the modal.
  return new Promise((resolve) => {
    const checkInterval = setInterval(async () => {
      const nowReady = await invoke<boolean>("is_model_ready", { name: modelName });
      if (nowReady) {
        clearInterval(checkInterval);
        modelModal.classList.add("hidden");
        resolve(true);
      }
    }, 1000);

    // Also resolve false if modal is closed without downloading
    const observer = new MutationObserver(() => {
      if (modelModal.classList.contains("hidden")) {
        observer.disconnect();
        clearInterval(checkInterval);
        invoke<boolean>("is_model_ready", { name: modelName }).then(resolve);
      }
    });
    observer.observe(modelModal, { attributes: true, attributeFilter: ["class"] });

    setTimeout(() => { clearInterval(checkInterval); observer.disconnect(); resolve(false); }, 30 * 60 * 1000);
  });
}

async function runChapterDetection(item: QueueItem, skipCueEmbed = false): Promise<void> {
  const settings = loadSettings();
  if (!settings.apiKey) {
    item.error = (item.error || "") + " (Set an OpenRouter API key in Settings to detect chapters)";
    return;
  }

  item.status = "detecting";
  renderQueue();

  const outputDir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
  const stem = item.name.replace(/\.[^.]+$/, "");
  const model = item.modelUsed || selectModel.value;
  const srtPath = `${outputDir}/${stem}_transcription_${model}.srt`;

  try {
    const transcript = await invoke<string>("read_text_file", { path: srtPath });
    const useSnap = chkSnapGaps.checked;
    let chapters: Chapter[];

    if (useSnap) {
      const snappedChapters = await invoke<ChapterWithSnap[]>("detect_chapters_with_gaps", {
        req: {
          transcript,
          api_key: settings.apiKey,
          model: settings.llmModel,
          base_url: settings.apiUrl,
          prompt: settings.chapterPrompt,
          transcript_path: srtPath, raw_mode: !chkSrtCorrect.checked,
        },
        audioPath: item.path,
        minGapSecs: 1.5,
        silenceThreshold: 0.02,
        maxLookbackSecs: 60.0,
      });
      item.snappedChapters = snappedChapters;
      chapters = snappedChapters.map((ch) => ({
        title: ch.title,
        start_time: ch.start_time,
        start_secs: ch.start_secs,
      }));
    } else {
      chapters = await invoke<Chapter[]>("detect_chapters", {
        req: {
          transcript,
          api_key: settings.apiKey,
          model: settings.llmModel,
          base_url: settings.apiUrl,
          prompt: settings.chapterPrompt,
          transcript_path: srtPath, raw_mode: !chkSrtCorrect.checked,
        },
      });
    }

    // Force first chapter to 0:00 if checkbox is checked
    if (chkFirstZero.checked && chapters.length > 0 && chapters[0].start_secs > 0) {
      chapters[0].start_secs = 0;
      chapters[0].start_time = "00:00:00";
    }

    item.chapters = chapters;

    if (chapters.length > 0) {
      const chapterFormat = settings.chapterOutputFormat || "txt";
      let content: string;
      let ext: string;

      if (chapterFormat === "json") {
        content = JSON.stringify(chapters, null, 2);
        ext = "json";
      } else {
        content = chapters.map((ch) => `${ch.start_time} - ${ch.title}`).join("\n");
        ext = "txt";
      }

      const llmShort = llmModelShort(settings.llmModel);
      const chapterPath = `${outputDir}/${stem}_chapters_${llmShort}.${ext}`;
      await invoke("write_text_file", { path: chapterPath, content });

      // Embed chapters in FLAC and write .cue (skip during reprocess to preserve manual alignment)
      if (!skipCueEmbed) {
        if (chkEmbedFlac.checked) {
          try {
            await invoke("embed_chapters_in_flac", { req: { audio_path: item.path, chapters } });
          } catch (embedErr) { console.warn("Embed chapters failed:", embedErr); }
        }
        if (chkWriteCue.checked) {
          try {
            await invoke("write_cue_file", { audioPath: item.path, chapters });
          } catch (cueErr) { console.warn("Write cue failed:", cueErr); }
        }
      }
    }
  } catch (err: any) {
    const errMsg = typeof err === "string" ? err : err?.message || "Chapter detection failed";
    item.error = (item.error || "") + ` (Chapters: ${errMsg})`;
  }
}

interface AssemblyAIResult {
  transcript_id: string;
  srt_path: string;
  json_path: string;
  txt_path: string;
  speaker_count: number;
  duration_secs: number | null;
}

async function transcribeItemAssemblyAI(item: QueueItem) {
  item.modelUsed = "AssemblyAI";
  item.error = undefined;
  renderQueue();

  const settings = loadSettings();
  if (!settings.assemblyaiKey) {
    item.status = "error";
    item.error = "AssemblyAI API key is not set. Open Settings to add it.";
    renderQueue();
    return;
  }

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    const alreadyExists = await checkOutputExists("check_diarization_exists", {
      path: item.path,
      engine: "assemblyai",
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      const stem = item.path.replace(/\.[^./]+$/, "").split("/").pop() || "";
      const dir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      item.diarizedSrtPath = `${dir}/${stem}.diarized.assemblyai.srt`;
      item.diarizedJsonPath = `${dir}/${stem}.diarized.assemblyai.json`;
      item.diarizedTxtPath = `${dir}/${stem}.diarized.assemblyai.txt`;
      item.status = "complete";
      item.progress = 1.0;
      renderQueue();
      return;
    }

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    const langTrimmed = (settings.assemblyaiLang || "").trim();
    const result = await invoke<AssemblyAIResult>("transcribe_assemblyai", {
      job: {
        id: item.id,
        path: item.path,
        api_key: settings.assemblyaiKey,
        output_dir: customOutputDir,
        language_code: langTrimmed || null,
        speech_models: assemblyaiSpeechModels(settings.assemblyaiModel),
        speakers_expected: settings.speakersExpected,
        trim_silence: settings.assemblyaiTrimSilence,
        silence_threshold_db: settings.assemblyaiSilenceDb,
        min_silence_secs: settings.assemblyaiMinSilenceSecs,
        silence_padding_secs: settings.assemblyaiSilencePadSecs,
      },
    });

    item.speakerCount = result.speaker_count;
    item.diarizedSrtPath = result.srt_path;
    item.diarizedJsonPath = result.json_path;
    item.diarizedTxtPath = result.txt_path;
    item.elapsed = Date.now() - (item.startedAt || Date.now());

    if ((item.status as string) !== "cancelled") {
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();

  if (item.status === "complete" && item.engine === "assemblyai" && item.diarizedJsonPath) {
    const modal = document.getElementById("speakers-modal");
    if (modal && modal.classList.contains("hidden")) {
      openSpeakerModal(item);
    }
  }
}

interface AAIUtterance {
  start: number;
  end: number;
  text: string;
  speaker: string;
}

interface SpeakerSample {
  speaker: string;
  count: number;
  totalSecs: number;
  samples: { text: string; start: number; secs: number }[];
}

function summarizeSpeakers(utterances: AAIUtterance[]): SpeakerSample[] {
  const byKey = new Map<string, AAIUtterance[]>();
  for (const u of utterances) {
    const key = u.speaker;
    if (!byKey.has(key)) byKey.set(key, []);
    byKey.get(key)!.push(u);
  }
  const out: SpeakerSample[] = [];
  for (const [speaker, group] of byKey) {
    const totalSecs = group.reduce((acc, u) => acc + Math.max(0, (u.end - u.start) / 1000), 0);
    // Pick up to 3 longest utterances for context
    const sorted = [...group].sort((a, b) => (b.end - b.start) - (a.end - a.start));
    const samples = sorted.slice(0, 3).map((u) => ({
      text: u.text.trim(),
      start: u.start,
      secs: (u.end - u.start) / 1000,
    }));
    out.push({ speaker, count: group.length, totalSecs, samples });
  }
  out.sort((a, b) => a.speaker.localeCompare(b.speaker));
  return out;
}

interface SpeakerModalState {
  item: QueueItem | null;          // null in standalone mode
  utterances: AAIUtterance[];
  raw: any | null;                 // present when source is JSON
  prefixIsLiteral: boolean;        // false: "Speaker A:", true: any "<Label>:"
  paths: { srtPath?: string; txtPath?: string; jsonPath?: string };
}

let speakerModalState: SpeakerModalState | null = null;

function showSpeakerModal(state: SpeakerModalState, prefilledNames?: Record<string, string>) {
  speakerModalState = state;
  const summary = summarizeSpeakers(state.utterances);

  const list = document.getElementById("speakers-list")!;
  list.innerHTML = summary.map((s) => {
    const heading = state.prefixIsLiteral ? escapeHtml(s.speaker) : `Speaker ${escapeHtml(s.speaker)}`;
    const prefilled = prefilledNames?.[s.speaker] || "";
    return `
      <div class="speaker-block">
        <div class="speaker-block-head">
          <span class="speaker-tag">${heading}</span>
          <span class="speaker-stats">${s.count} utterances · ${formatDuration(s.totalSecs)} total</span>
        </div>
        <div class="speaker-samples">
          ${s.samples.map((sm) => `<div class="speaker-sample"><span class="sample-time">${formatDuration(sm.start / 1000)}</span> <span class="sample-text">${escapeHtml(sm.text)}</span></div>`).join("")}
        </div>
        <label class="speaker-name-row">
          Name:
          <input type="text" class="form-input speaker-name-input" data-speaker="${escapeHtml(s.speaker)}" value="${escapeHtml(prefilled)}" placeholder="e.g. Narrator" />
        </label>
      </div>
    `;
  }).join("");

  document.getElementById("speakers-modal")!.classList.remove("hidden");
}

async function openSpeakerModal(item: QueueItem) {
  if (!item.diarizedJsonPath) {
    alert("No diarized JSON file found for this item.");
    return;
  }
  let raw: any;
  try {
    const content = await invoke<string>("read_text_file", { path: item.diarizedJsonPath });
    raw = JSON.parse(content);
  } catch (err) {
    alert(`Failed to load diarized JSON: ${err}`);
    return;
  }
  const utterances: AAIUtterance[] = raw.utterances || [];
  if (utterances.length === 0) {
    alert("No utterances found in diarized JSON.");
    return;
  }
  showSpeakerModal({
    item,
    utterances,
    raw,
    prefixIsLiteral: false, // queue items use "Speaker A" naming
    paths: {
      srtPath: item.diarizedSrtPath,
      txtPath: item.diarizedTxtPath,
      jsonPath: item.diarizedJsonPath,
    },
  }, item.speakerNames);
}

function siblingPaths(srcPath: string): { srtPath?: string; txtPath?: string; jsonPath?: string } {
  const stem = srcPath.replace(/\.(srt|txt|json)$/i, "");
  return {
    srtPath: stem + ".srt",
    txtPath: stem + ".txt",
    jsonPath: stem + ".json",
  };
}

function parseDiarizedSrt(text: string): { utterances: AAIUtterance[] } {
  const utterances: AAIUtterance[] = [];
  const blocks = text.split(/\r?\n\r?\n+/);
  for (const block of blocks) {
    const lines = block.split(/\r?\n/).filter((l) => l.length > 0);
    if (lines.length < 2) continue;
    let timeIdx = 0;
    if (/^\d+$/.test(lines[0].trim())) timeIdx = 1;
    const timeLine = lines[timeIdx];
    const m = timeLine.match(/(\d{1,2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2},\d{3})/);
    if (!m) continue;
    const start = srtTimeToMs(m[1]);
    const end = srtTimeToMs(m[2]);
    const textLines = lines.slice(timeIdx + 1).join(" ").trim();
    const colonIdx = textLines.indexOf(":");
    if (colonIdx < 1 || colonIdx > 60) continue; // no speaker prefix recognized
    const speaker = textLines.substring(0, colonIdx).trim();
    const text = textLines.substring(colonIdx + 1).trim();
    utterances.push({ start, end, text, speaker });
  }
  return { utterances };
}

function parseDiarizedTxt(text: string): { utterances: AAIUtterance[] } {
  const utterances: AAIUtterance[] = [];
  const blocks = text.split(/\r?\n\r?\n+/);
  let cursor = 0;
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    const colonIdx = trimmed.indexOf(":");
    if (colonIdx < 1 || colonIdx > 60) continue;
    const speaker = trimmed.substring(0, colonIdx).trim();
    const utteranceText = trimmed.substring(colonIdx + 1).trim();
    // No timestamps in TXT — fabricate increasing range so summary ordering works
    utterances.push({ start: cursor, end: cursor + 1000, text: utteranceText, speaker });
    cursor += 1000;
  }
  return { utterances };
}

function srtTimeToMs(t: string): number {
  const m = t.match(/(\d{1,2}):(\d{2}):(\d{2}),(\d{3})/);
  if (!m) return 0;
  return parseInt(m[1]) * 3600000 + parseInt(m[2]) * 60000 + parseInt(m[3]) * 1000 + parseInt(m[4]);
}

async function openStandaloneSpeakerModal(filePath: string) {
  const lower = filePath.toLowerCase();
  let utterances: AAIUtterance[] = [];
  let raw: any | null = null;
  let prefixIsLiteral = true;

  try {
    const content = await invoke<string>("read_text_file", { path: filePath });
    if (lower.endsWith(".json")) {
      raw = JSON.parse(content);
      utterances = raw.utterances || [];
      prefixIsLiteral = false; // JSON keys are still raw "A", "B" speaker codes
    } else if (lower.endsWith(".srt")) {
      ({ utterances } = parseDiarizedSrt(content));
    } else if (lower.endsWith(".txt")) {
      ({ utterances } = parseDiarizedTxt(content));
    } else {
      alert("Please pick a .srt, .txt, or .json file.");
      return;
    }
  } catch (err) {
    alert(`Failed to read file: ${err}`);
    return;
  }

  if (utterances.length === 0) {
    alert("No speaker-labelled utterances found in this file.");
    return;
  }

  const sib = siblingPaths(filePath);
  // Only target paths that actually exist
  const exists = await Promise.all([
    sib.srtPath ? invoke<boolean>("file_exists", { path: sib.srtPath }).catch(() => false) : Promise.resolve(false),
    sib.txtPath ? invoke<boolean>("file_exists", { path: sib.txtPath }).catch(() => false) : Promise.resolve(false),
    sib.jsonPath ? invoke<boolean>("file_exists", { path: sib.jsonPath }).catch(() => false) : Promise.resolve(false),
  ]);
  const paths = {
    srtPath: exists[0] ? sib.srtPath : undefined,
    txtPath: exists[1] ? sib.txtPath : undefined,
    jsonPath: exists[2] ? sib.jsonPath : undefined,
  };

  // Pre-fill from JSON's speaker_names map if present
  const prefilled: Record<string, string> = (raw && raw.speaker_names) || {};

  showSpeakerModal({
    item: null,
    utterances,
    raw,
    prefixIsLiteral,
    paths,
  }, prefilled);
}

function closeSpeakerModal() {
  document.getElementById("speakers-modal")!.classList.add("hidden");
  speakerModalState = null;
}

async function saveSpeakerNames() {
  if (!speakerModalState) return;
  const { item, utterances, raw, prefixIsLiteral, paths } = speakerModalState;

  const inputs = document.querySelectorAll<HTMLInputElement>(".speaker-name-input");
  const names: Record<string, string> = {};
  inputs.forEach((inp) => {
    const k = inp.dataset.speaker!;
    const v = inp.value.trim();
    if (v) names[k] = v;
  });

  const labelFor = (speaker: string) => {
    if (names[speaker]) return names[speaker];
    return prefixIsLiteral ? speaker : `Speaker ${speaker}`;
  };

  // Rewrite SRT (only if we have real timestamps — i.e. not fabricated by TXT parsing)
  const haveTimestamps = utterances.some((u) => u.end > u.start && u.end > 0);
  let srt: string | null = null;
  if (haveTimestamps) {
    const srtLines: string[] = [];
    utterances.forEach((u, i) => {
      srtLines.push(String(i + 1));
      srtLines.push(`${msToSrt(u.start)} --> ${msToSrt(u.end)}`);
      srtLines.push(`${labelFor(u.speaker)}: ${u.text.trim()}`);
      srtLines.push("");
    });
    srt = srtLines.join("\n");
  }

  const txt = utterances.map((u) => `${labelFor(u.speaker)}: ${u.text.trim()}`).join("\n\n") + "\n";

  let json: string | null = null;
  if (raw) {
    const updated = { ...raw, speaker_names: names };
    if (updated.utterances) {
      updated.utterances = updated.utterances.map((u: any) => ({
        ...u,
        speaker_name: names[u.speaker] || null,
      }));
    }
    json = JSON.stringify(updated, null, 2);
  }

  try {
    if (paths.srtPath && srt !== null) await invoke("write_text_file", { path: paths.srtPath, content: srt });
    if (paths.txtPath) await invoke("write_text_file", { path: paths.txtPath, content: txt });
    if (paths.jsonPath && json !== null) await invoke("write_text_file", { path: paths.jsonPath, content: json });
  } catch (err) {
    alert(`Failed to write files: ${err}`);
    return;
  }

  if (item) {
    item.speakerNames = names;
    renderQueue();
  }
  closeSpeakerModal();
}

function msToSrt(ms: number): string {
  const totalSecs = Math.floor(ms / 1000);
  const h = Math.floor(totalSecs / 3600);
  const m = Math.floor((totalSecs % 3600) / 60);
  const s = totalSecs % 60;
  const mmm = Math.floor(ms % 1000);
  return `${pad2(h)}:${pad2(m)}:${pad2(s)},${pad3(mmm)}`;
}
function pad2(n: number): string { return n.toString().padStart(2, "0"); }
function pad3(n: number): string { return n.toString().padStart(3, "0"); }

async function transcribeItemSherpa(item: QueueItem) {
  const model = selectModel.value;
  const threads = parseInt(selectThreads.value);
  item.modelUsed = `${model} + Sherpa`;
  item.error = undefined;
  renderQueue();

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    const alreadyExists = await checkOutputExists("check_diarization_exists", {
      path: item.path,
      engine: "sherpa",
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      const stem = item.path.replace(/\.[^./]+$/, "").split("/").pop() || "";
      const dir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      item.diarizedSrtPath = `${dir}/${stem}.diarized.sherpa.srt`;
      item.diarizedJsonPath = `${dir}/${stem}.diarized.sherpa.json`;
      item.diarizedTxtPath = `${dir}/${stem}.diarized.sherpa.txt`;
      item.status = "complete";
      item.progress = 1.0;
      renderQueue();
      return;
    }

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    const settings = loadSettings();
    const result = await invoke<AssemblyAIResult>("transcribe_sherpa", {
      job: {
        id: item.id,
        path: item.path,
        model,
        output_dir: customOutputDir,
        threads,
        num_speakers: settings.sherpaNumSpeakers,
        threshold: settings.sherpaThreshold,
      },
    });

    item.speakerCount = result.speaker_count;
    item.diarizedSrtPath = result.srt_path;
    item.diarizedJsonPath = result.json_path;
    item.diarizedTxtPath = result.txt_path;
    item.elapsed = Date.now() - (item.startedAt || Date.now());

    if ((item.status as string) !== "cancelled") {
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();

  if (item.status === "complete" && item.diarizedJsonPath) {
    const modal = document.getElementById("speakers-modal");
    if (modal && modal.classList.contains("hidden")) {
      openSpeakerModal(item);
    }
  }
}

async function transcribeItemDeepgram(item: QueueItem) {
  item.modelUsed = "Deepgram";
  item.error = undefined;
  renderQueue();

  const settings = loadSettings();
  if (!settings.deepgramKey) {
    item.status = "error";
    item.error = "Deepgram API key is not set. Open Settings to add it.";
    renderQueue();
    return;
  }

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    const alreadyExists = await checkOutputExists("check_diarization_exists", {
      path: item.path,
      engine: "deepgram",
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      const stem = item.path.replace(/\.[^./]+$/, "").split("/").pop() || "";
      const dir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      item.diarizedSrtPath = `${dir}/${stem}.diarized.deepgram.srt`;
      item.diarizedJsonPath = `${dir}/${stem}.diarized.deepgram.json`;
      item.diarizedTxtPath = `${dir}/${stem}.diarized.deepgram.txt`;
      item.status = "complete";
      item.progress = 1.0;
      renderQueue();
      return;
    }

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    const langTrimmed = (settings.deepgramLang || "").trim();
    const result = await invoke<AssemblyAIResult>("transcribe_deepgram", {
      job: {
        id: item.id,
        path: item.path,
        api_key: settings.deepgramKey,
        model: settings.deepgramModel,
        output_dir: customOutputDir,
        language_code: langTrimmed || null,
      },
    });

    item.speakerCount = result.speaker_count;
    item.diarizedSrtPath = result.srt_path;
    item.diarizedJsonPath = result.json_path;
    item.diarizedTxtPath = result.txt_path;
    item.elapsed = Date.now() - (item.startedAt || Date.now());

    if ((item.status as string) !== "cancelled") {
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();

  if (item.status === "complete" && item.diarizedJsonPath) {
    const modal = document.getElementById("speakers-modal");
    if (modal && modal.classList.contains("hidden")) {
      openSpeakerModal(item);
    }
  }
}

async function transcribeItemParakeet(item: QueueItem) {
  const format = selectFormat.value;
  const threads = parseInt(selectThreads.value);
  item.modelUsed = "Parakeet";
  item.error = undefined;
  renderQueue();

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    // Parakeet writes the same {stem}_transcription_parakeet.{ext} files Whisper
    // does, so the standard transcription-exists check works with model "parakeet".
    const alreadyExists = await checkOutputExists("check_transcription_exists", {
      path: item.path,
      model: "parakeet",
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      item.status = "complete";
      item.progress = 1.0;
      renderQueue();
      return;
    }

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    await invoke("transcribe_parakeet", {
      job: {
        id: item.id,
        path: item.path,
        output_format: format,
        output_dir: customOutputDir,
        threads,
      },
    });

    item.elapsed = Date.now() - (item.startedAt || Date.now());

    if ((item.status as string) !== "cancelled") {
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();
}

async function transcribeItemParakeetSherpa(item: QueueItem) {
  const threads = parseInt(selectThreads.value);
  item.modelUsed = "Parakeet + Sherpa";
  item.error = undefined;
  renderQueue();

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    const alreadyExists = await checkOutputExists("check_diarization_exists", {
      path: item.path,
      engine: "parakeet-sherpa",
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      const stem = item.path.replace(/\.[^./]+$/, "").split("/").pop() || "";
      const dir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      item.diarizedSrtPath = `${dir}/${stem}.diarized.parakeet-sherpa.srt`;
      item.diarizedJsonPath = `${dir}/${stem}.diarized.parakeet-sherpa.json`;
      item.diarizedTxtPath = `${dir}/${stem}.diarized.parakeet-sherpa.txt`;
      item.status = "complete";
      item.progress = 1.0;
      renderQueue();
      return;
    }

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    const settings = loadSettings();
    const result = await invoke<AssemblyAIResult>("transcribe_parakeet_sherpa", {
      job: {
        id: item.id,
        path: item.path,
        output_dir: customOutputDir,
        threads,
        num_speakers: settings.sherpaNumSpeakers,
        threshold: settings.sherpaThreshold,
      },
    });

    item.speakerCount = result.speaker_count;
    item.diarizedSrtPath = result.srt_path;
    item.diarizedJsonPath = result.json_path;
    item.diarizedTxtPath = result.txt_path;
    item.elapsed = Date.now() - (item.startedAt || Date.now());

    if ((item.status as string) !== "cancelled") {
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();

  if (item.status === "complete" && item.diarizedJsonPath) {
    const modal = document.getElementById("speakers-modal");
    if (modal && modal.classList.contains("hidden")) {
      openSpeakerModal(item);
    }
  }
}

// Run a file through every downloaded local transcription model (each Whisper
// size on disk + Parakeet if ready), sequentially, for manual side-by-side
// comparison. Each model writes its own {stem}_transcription_{model}.{ext}, so
// nothing collides. Honours the Format dropdown and Force-overwrite.
// Build the Markdown comparison summary for "Compare all local models".
function buildCompareMarkdown(
  fileName: string,
  audioDuration: number | null,
  format: string,
  rows: { label: string; model: string; seconds: number | null; words: number | null; chars: number | null; preview: string }[],
): string {
  const clock = (s: number) => {
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    return h > 0
      ? `${h}:${m.toString().padStart(2, "0")}:${sec.toString().padStart(2, "0")}`
      : `${m}:${sec.toString().padStart(2, "0")}`;
  };
  const dur = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m ${Math.round(s % 60)}s` : `${s.toFixed(1)}s`);

  let md = `# Local model comparison\n\n`;
  md += `**File:** ${fileName}  \n`;
  if (audioDuration) md += `**Audio length:** ${clock(audioDuration)} (${audioDuration.toFixed(0)}s)  \n`;
  md += `**Generated:** ${new Date().toLocaleString()}\n\n`;

  md += `| Model | Time | Speed | Words | Chars |\n`;
  md += `|---|---|---|---|---|\n`;
  for (const r of rows) {
    if (r.seconds == null) {
      md += `| ${r.label} | — (already on disk) | — | — | — |\n`;
      continue;
    }
    const speed = audioDuration && r.seconds > 0 ? `${(audioDuration / r.seconds).toFixed(1)}× realtime` : "—";
    md += `| ${r.label} | ${dur(r.seconds)} | ${speed} | ${r.words?.toLocaleString() ?? "—"} | ${r.chars?.toLocaleString() ?? "—"} |\n`;
  }

  const skipped = rows.filter((r) => r.seconds == null).length;
  if (skipped > 0) {
    md += `\n> ${skipped} model(s) were already transcribed on disk and were not re-run, so they have no timing. Tick **Force overwrite** and run again for a clean head-to-head timing comparison.\n`;
  }

  const withText = rows.filter((r) => r.preview);
  if (withText.length > 0) {
    md += `\n## Transcript previews (first ~280 chars)\n`;
    for (const r of withText) {
      md += `\n### ${r.label}\n\n> ${r.preview.replace(/\s+/g, " ").trim()}…\n`;
    }
  }

  md += `\n---\nFull transcripts: \`<file>_transcription_<model>.${format === "all" ? "{txt,srt,vtt,json}" : format}\` in this folder.\n`;
  return md;
}

async function transcribeItemCompare(item: QueueItem) {
  const format = selectFormat.value;
  const threads = parseInt(selectThreads.value);
  item.error = undefined;
  renderQueue();

  // Discover what's available locally.
  const allModels = await invoke<ModelInfo[]>("list_models");
  const parakeetReady = await invoke<boolean>("parakeet_model_ready");

  type Run = { kind: "whisper" | "parakeet"; model: string; label: string };
  const runs: Run[] = allModels
    .filter((m) => m.downloaded)
    .map((m) => ({ kind: "whisper" as const, model: m.name, label: m.name }));
  if (parakeetReady) runs.push({ kind: "parakeet", model: "parakeet", label: "Parakeet" });

  if (runs.length === 0) {
    item.status = "error";
    item.error =
      "No local transcription models are downloaded. Download Whisper sizes and/or Parakeet from the Models manager first.";
    renderQueue();
    return;
  }

  // Per-model timing/quality, collected for the comparison summary.
  type CompareRow = {
    label: string;
    model: string;
    seconds: number | null; // null = skipped (already on disk)
    words: number | null;
    chars: number | null;
    preview: string;
  };
  const rows: CompareRow[] = [];

  try {
    if (item.status === "cancelled" || !queue.includes(item)) return;

    item.status = "queued";
    item.progress = 0;
    item.startedAt = undefined;
    item.elapsed = undefined;
    renderQueue();

    const startAll = Date.now();
    let audioDuration = item.duration || null;
    let done = 0;
    for (const run of runs) {
      if ((item.status as string) === "cancelled" || !queue.includes(item)) break;
      // Surface the current model in the model column (the progress listener
      // overwrites stageText, but never touches modelUsed).
      item.modelUsed = `${run.label} (${done + 1}/${runs.length})`;
      renderQueue();

      const exists = await checkOutputExists("check_transcription_exists", {
        path: item.path,
        model: run.model,
        outputDir: customOutputDir || null,
      });
      if (exists) {
        rows.push({ label: run.label, model: run.model, seconds: null, words: null, chars: null, preview: "" });
        done++;
        continue;
      }

      const t0 = Date.now();
      const result =
        run.kind === "whisper"
          ? await invoke<TranscriptionResult>("transcribe_file", {
              job: {
                id: item.id,
                path: item.path,
                model: run.model,
                output_format: format,
                output_dir: customOutputDir,
                threads,
                per_word: false,
              },
            })
          : await invoke<TranscriptionResult>("transcribe_parakeet", {
              job: {
                id: item.id,
                path: item.path,
                output_format: format,
                output_dir: customOutputDir,
                threads,
              },
            });
      const seconds = (Date.now() - t0) / 1000;
      if (!audioDuration && result?.duration_secs) audioDuration = result.duration_secs;
      const text = (result?.text || "").trim();
      const words = text ? text.split(/\s+/).length : 0;
      rows.push({
        label: run.label,
        model: run.model,
        seconds,
        words,
        chars: text.length,
        preview: text.slice(0, 280),
      });
      console.log(`[compare] ${run.label}: ${seconds.toFixed(1)}s, ${words} words`);
      done++;
    }

    // Write the comparison summary (overwrites on each run).
    if ((item.status as string) !== "cancelled" && rows.length > 0) {
      const outputDir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
      const stem = item.name.replace(/\.[^.]+$/, "");
      const md = buildCompareMarkdown(item.name, audioDuration, format, rows);
      try {
        await invoke("write_text_file", { path: `${outputDir}/${stem}_compare.md`, content: md });
      } catch (e) {
        console.warn("Failed to write comparison summary:", e);
      }
    }

    item.elapsed = Date.now() - startAll;
    if ((item.status as string) !== "cancelled") {
      item.modelUsed = `${runs.length} models compared`;
      item.status = "complete";
      item.progress = 1.0;
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();
}

async function transcribeItem(item: QueueItem) {
  if (item.engine === "compare-local") {
    await transcribeItemCompare(item);
    return;
  }
  if (item.engine === "assemblyai") {
    await transcribeItemAssemblyAI(item);
    return;
  }
  if (item.engine === "deepgram") {
    await transcribeItemDeepgram(item);
    return;
  }
  if (item.engine === "sherpa") {
    await transcribeItemSherpa(item);
    return;
  }
  if (item.engine === "parakeet") {
    await transcribeItemParakeet(item);
    return;
  }
  if (item.engine === "parakeet-sherpa") {
    await transcribeItemParakeetSherpa(item);
    return;
  }

  const model = selectModel.value;
  const format = selectFormat.value;
  const threads = parseInt(selectThreads.value);

  item.modelUsed = model;
  item.error = undefined;
  renderQueue();

  try {
    // Check if already cancelled or removed before starting
    if (item.status === "cancelled" || !queue.includes(item)) return;

    // Check if transcription already exists for this model
    const alreadyExists = await checkOutputExists("check_transcription_exists", {
      path: item.path,
      model,
      outputDir: customOutputDir || null,
    });

    if (alreadyExists) {
      // Skip transcription, but still run chapter detection if needed
      if (item.autoDetectChapters && (item.status as string) !== "cancelled" && queue.includes(item)) {
        item.status = "detecting";
        item.progress = 1.0;
        renderQueue();
        await runChapterDetection(item);
      }
      if ((item.status as string) !== "cancelled" && queue.includes(item)) {
        item.status = "complete";
        item.progress = 1.0;
      }
    } else {
      item.status = "queued";
      item.progress = 0;
      item.startedAt = undefined;
      item.elapsed = undefined;
      renderQueue();

      // If chapters needed, ensure SRT is produced alongside selected format
      let outputFormat = format;
      if (item.autoDetectChapters && format !== "all" && format !== "srt") {
        outputFormat = `${format},srt`;
      }

      await invoke("transcribe_file", {
        job: {
          id: item.id,
          path: item.path,
          model,
          output_format: outputFormat,
          output_dir: customOutputDir,
          threads,
          per_word: chkPerWord.checked,
        },
      });

      item.elapsed = Date.now() - (item.startedAt || Date.now());

      // Auto-detect chapters if enabled (check cancel/removed before starting)
      if (item.autoDetectChapters && (item.status as string) !== "cancelled" && queue.includes(item)) {
        await runChapterDetection(item);
      }

      if ((item.status as string) !== "cancelled") {
        item.status = "complete";
        item.progress = 1.0;
      }
    }
  } catch (err: any) {
    item.elapsed = item.startedAt ? Date.now() - item.startedAt : undefined;
    const errMsg = typeof err === "string" ? err : err?.message || "Unknown error";
    if (errMsg === "Cancelled") {
      item.status = "cancelled";
      item.error = "Cancelled by user";
    } else {
      item.status = "error";
      item.error = errMsg;
    }
  }
  renderQueue();
}

async function transcribeAll() {
  const model = selectModel.value;
  const pendingItems = queue.filter((q) => q.status === "pending");
  if (pendingItems.length === 0) return;

  // Reconcile queued engines against the current dropdown. Items are stamped
  // with the engine selected at add-time; if the dropdown now says something
  // else, the user almost certainly means "run these with the selected engine".
  // Offer to re-stamp before doing anything (esp. important so a local file is
  // never silently sent to a cloud engine, or vice-versa).
  const selectedEngine = (selectEngine.value as Engine) || "whisper";
  const mismatched = pendingItems.filter((i) => i.engine !== selectedEngine);
  if (mismatched.length > 0) {
    const fromLabel = engineShort(mismatched[0].engine);
    const allSameFrom = mismatched.every((i) => i.engine === mismatched[0].engine);
    const fromText = allSameFrom ? fromLabel : "a different engine";
    const switchThem = await ask(
      `${mismatched.length} pending file(s) were queued with ${fromText}, but ` +
      `${engineShort(selectedEngine)} is now selected.\n\n` +
      `Switch them to ${engineShort(selectedEngine)}?\n\n` +
      `(Yes = run everything with ${engineShort(selectedEngine)}. ` +
      `No = run each file with the engine it was queued with.)`,
      { title: "Engine mismatch", kind: "warning" }
    );
    if (switchThem) {
      mismatched.forEach((i) => {
        i.engine = selectedEngine;
        // Chapter detection is a Whisper-only feature; clear it otherwise.
        i.autoDetectChapters = selectedEngine === "whisper" && chkAutoChapters.checked;
      });
      renderQueue();
    }
  }

  const whisperItems = pendingItems.filter((i) => i.engine === "whisper");
  const aaiItems = pendingItems.filter((i) => i.engine === "assemblyai");
  const dgItems = pendingItems.filter((i) => i.engine === "deepgram");
  const sherpaItems = pendingItems.filter((i) => i.engine === "sherpa");
  const parakeetItems = pendingItems.filter((i) => i.engine === "parakeet");
  const parakeetSherpaItems = pendingItems.filter((i) => i.engine === "parakeet-sherpa");

  // Sherpa needs the Whisper model too
  if (whisperItems.length > 0 || sherpaItems.length > 0) {
    const modelReady = await checkModelAndPromptDownload(model);
    if (!modelReady) return;
  }

  // Parakeet (alone or paired with Sherpa) uses its own model — no Whisper dep.
  if (parakeetItems.length > 0 || parakeetSherpaItems.length > 0) {
    const ready = await invoke<boolean>("parakeet_model_ready");
    if (!ready) {
      const doDownload = confirm(
        "The Parakeet model is not downloaded yet (~482 MB, plus a small Silero VAD model).\n\nDownload it now?"
      );
      if (!doDownload) return;
      try {
        await invoke("download_parakeet_model");
      } catch (err) {
        alert(`Failed to download the Parakeet model: ${err}`);
        return;
      }
    }
  }

  // Parakeet+Sherpa also needs the Sherpa diarization models.
  if (parakeetSherpaItems.length > 0 || sherpaItems.length > 0) {
    const ready = await invoke<boolean>("sherpa_models_ready");
    if (!ready) {
      const doDownload = confirm(
        "Sherpa-onnx diarization models are not downloaded yet (~28 MB total: pyannote segmentation + speaker embedding).\n\nDownload them now?"
      );
      if (!doDownload) return;
      try {
        await invoke("download_sherpa_models");
      } catch (err) {
        alert(`Failed to download Sherpa models: ${err}`);
        return;
      }
    }
  }

  const settings = loadSettings();
  if (aaiItems.length > 0) {
    if (!settings.assemblyaiKey) {
      alert("AssemblyAI API key is not set. Open Settings to add it before submitting AssemblyAI jobs.");
      return;
    }
    if (!(await confirmAssemblyAICost(aaiItems))) return;
  }
  if (dgItems.length > 0) {
    if (!settings.deepgramKey) {
      alert("Deepgram API key is not set. Open Settings to add it before submitting Deepgram jobs.");
      return;
    }
    if (!(await confirmDeepgramCost(dgItems))) return;
  }

  btnTranscribeAll.disabled = true;

  const concurrent = parseInt(selectConcurrent.value);
  await invoke("set_concurrency", { permits: concurrent });

  // Update autoDetectChapters based on current checkbox state (Whisper items only)
  whisperItems.forEach((item) => {
    item.autoDetectChapters = chkAutoChapters.checked;
  });

  const promises = pendingItems.map((item) => transcribeItem(item));
  await Promise.all(promises);

  btnTranscribeAll.disabled = false;
}

function assemblyaiRatePerHour(modelChoice: string): { rate: number; label: string } {
  // AssemblyAI: U2 = $0.15/hr, U3-Pro = $0.21/hr, +$0.02/hr diarization add-on
  switch (modelChoice) {
    case "universal-2":
      return { rate: 0.17, label: "Universal-2 + diarization" };
    case "universal-3-pro":
    case "auto":
    default:
      return { rate: 0.23, label: "Universal-3 Pro + diarization" };
  }
}

async function confirmDeepgramCost(items: QueueItem[]): Promise<boolean> {
  // Deepgram Nova-3 batch: $0.0043/min ≈ $0.258/hr (diarization included).
  const ratePerMin = 0.0043;
  const knownDurationSecs = items.map((i) => i.duration || 0).reduce((a, b) => a + b, 0);
  const unknownCount = items.filter((i) => !i.duration).length;
  const knownMinutes = knownDurationSecs / 60;
  const cost = knownMinutes * ratePerMin;

  let msg = `${items.length} file(s) will be sent to Deepgram (Nova-3, diarization included).\n\n`;
  if (knownDurationSecs > 0) {
    msg += `Total duration: ${formatDuration(knownDurationSecs)} (~${knownMinutes.toFixed(1)} min)\n`;
    msg += `Estimated cost: $${cost.toFixed(2)} (at $${(ratePerMin * 60).toFixed(2)}/hr)\n`;
  }
  if (unknownCount > 0) {
    msg += `${unknownCount} file(s) have unknown duration — final cost may be higher.\n`;
  }
  return await ask(msg, { title: "Confirm Deepgram cost", kind: "info", okLabel: "Send to Deepgram", cancelLabel: "Cancel" });
}

interface TrimPreviewResult {
  original_duration: number;
  trimmed_duration: number;
}

async function confirmAssemblyAICost(items: QueueItem[]): Promise<boolean> {
  const settings = loadSettings();
  const { rate: ratePerHour, label } = assemblyaiRatePerHour(settings.assemblyaiModel);
  const ratePerMin = ratePerHour / 60;

  let originalSecs = items.map((i) => i.duration || 0).reduce((a, b) => a + b, 0);
  const unknownCount = items.filter((i) => !i.duration).length;
  let billableSecs = originalSecs;
  let trimmedSomething = false;

  if (settings.assemblyaiTrimSilence) {
    for (let idx = 0; idx < items.length; idx++) {
      const item = items[idx];
      item.status = "queued";
      item.progress = 0;
      item.stageText = items.length > 1
        ? `analyzing silence (${idx + 1}/${items.length})…`
        : "analyzing silence…";
      renderQueue();
      try {
        const r = await invoke<TrimPreviewResult>("preview_silence_trim", {
          path: item.path,
          thresholdDb: settings.assemblyaiSilenceDb,
          minSilenceSecs: settings.assemblyaiMinSilenceSecs,
          paddingSecs: settings.assemblyaiSilencePadSecs,
        });
        billableSecs = billableSecs - (r.original_duration - r.trimmed_duration);
        trimmedSomething = true;
        item.stageText = `trim preview: ${formatDuration(r.original_duration)} → ${formatDuration(r.trimmed_duration)}`;
      } catch (err) {
        item.stageText = `trim preview failed: ${err}`;
      }
      item.status = "queued";
      renderQueue();
    }
  }

  const billableMinutes = billableSecs / 60;
  const cost = billableMinutes * ratePerMin;

  let msg = `${items.length} file(s) will be sent to AssemblyAI (${label}).\n\n`;
  if (originalSecs > 0) {
    msg += `Original duration: ${formatDuration(originalSecs)}\n`;
    if (trimmedSomething) {
      const savedSecs = originalSecs - billableSecs;
      const savedPct = originalSecs > 0 ? (savedSecs / originalSecs) * 100 : 0;
      msg += `After silence trim: ${formatDuration(billableSecs)} (~${billableMinutes.toFixed(1)} min, ${savedPct.toFixed(0)}% saved)\n`;
    } else {
      msg += `Total billable: ~${billableMinutes.toFixed(1)} min\n`;
    }
    msg += `Estimated cost: $${cost.toFixed(2)} (at $${ratePerHour.toFixed(2)}/hr)\n`;
  }
  if (unknownCount > 0) {
    msg += `${unknownCount} file(s) have unknown duration — final cost may be higher.\n`;
  }

  const proceed = await ask(msg, {
    title: "Confirm AssemblyAI cost",
    kind: "info",
    okLabel: "Send to AssemblyAI",
    cancelLabel: "Cancel",
  });
  if (!proceed) {
    try { await invoke("discard_all_trim_previews"); } catch {}
  }
  return proceed;
}

// GPU info
async function loadGpuInfo() {
  try {
    const gpu = await invoke<GpuInfo>("get_gpu_info");
    const dot = gpu.using_metal ? "active" : "inactive";
    const coresText = gpu.gpu_cores ? ` (${gpu.gpu_cores} cores)` : "";
    const metalText = gpu.using_metal ? "Metal GPU acceleration active" : "GPU not available";
    gpuBar.innerHTML = `<span class="gpu-dot ${dot}"></span> ${gpu.name}${coresText} &mdash; ${metalText}`;
  } catch {
    gpuBar.innerHTML = `<span class="gpu-dot inactive"></span> GPU detection failed`;
  }
}

// Model manager
interface SherpaModelInfo {
  name: string;
  display_name: string;
  downloaded: boolean;
  size_bytes: number;
  expected_bytes: number;
}

async function refreshModelList() {
  const models = await invoke<ModelInfo[]>("list_models");
  const dir = await invoke<string>("get_models_dir");
  const sherpaModels = await invoke<SherpaModelInfo[]>("list_sherpa_models");
  const parakeetModels = await invoke<SherpaModelInfo[]>("list_parakeet_models");
  modelsDir.textContent = dir;

  const whisperRow = (m: ModelInfo) => `
    <div class="model-item" data-model="${m.name}">
      <span class="model-name">${m.display_name}</span>
      ${
        m.downloaded
          ? `<span class="model-status downloaded">Downloaded</span>`
          : `<button class="small btn-download-model" data-name="${m.name}" data-kind="whisper">Download</button>`
      }
    </div>`;

  const sherpaRow = (m: SherpaModelInfo) => `
    <div class="model-item" data-model="${m.name}">
      <span class="model-name">${escapeHtml(m.display_name)}</span>
      ${
        m.downloaded
          ? `<span class="model-status downloaded">Downloaded</span>`
          : `<button class="small btn-download-model" data-name="${m.name}" data-kind="sherpa">Download</button>`
      }
    </div>`;

  const parakeetRow = (m: SherpaModelInfo) => `
    <div class="model-item" data-model="${m.name}">
      <span class="model-name">${escapeHtml(m.display_name)}</span>
      ${
        m.downloaded
          ? `<span class="model-status downloaded">Downloaded</span>`
          : `<button class="small btn-download-model" data-name="${m.name}" data-kind="parakeet">Download</button>`
      }
    </div>`;

  modelList.innerHTML = `
    <div class="model-section-title">Whisper Transcription Models</div>
    ${models.map(whisperRow).join("")}
    <div class="model-section-title">Parakeet (local) Transcription Model</div>
    ${parakeetModels.map(parakeetRow).join("")}
    <div class="model-section-title">Sherpa-onnx Diarization Models</div>
    ${sherpaModels.map(sherpaRow).join("")}
  `;

  modelList.querySelectorAll(".btn-download-model").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = (btn as HTMLElement).dataset.name!;
      const kind = (btn as HTMLElement).dataset.kind || "whisper";
      const item = btn.closest(".model-item")!;
      (btn as HTMLElement).outerHTML = `<div class="download-progress"><div class="progress-bar"><div class="fill model-dl-fill" data-model="${name}" style="width: 0%"></div></div><span class="model-dl-text" data-model="${name}">Starting...</span></div>`;

      try {
        const cmd =
          kind === "sherpa"
            ? "download_sherpa_model"
            : kind === "parakeet"
              ? "download_parakeet_model"
              : "download_model";
        await invoke(cmd, { name });
        const statusEl = item.querySelector(".download-progress");
        if (statusEl) statusEl.outerHTML = `<span class="model-status downloaded">Downloaded</span>`;
      } catch {
        const statusEl = item.querySelector(".download-progress");
        if (statusEl) statusEl.outerHTML = `<button class="small btn-download-model" data-name="${name}" data-kind="${kind}">Retry</button>`;
      }
    });
  });
}

// Feed
interface FeedEpisode {
  id: string;
  title: string;
  date: string;
  rawDate: string;
  audioUrl: string;
  status: "pending" | "downloading" | "complete" | "error";
  progress: number;
  downloadedMb: number;
  totalMb: number;
  localPath?: string;
}

let feedEpisodesList: FeedEpisode[] = [];
let feedSaveDir = "";

function feedUpdateSelectedCount() {
  const feedEpisodes = document.getElementById("feed-episodes")!;
  const count = feedEpisodes.querySelectorAll('input[type="checkbox"]:checked').length;
  document.getElementById("feed-selected-count")!.textContent = `${count} selected`;
}

function feedRenderEpisode(ep: FeedEpisode, idx: number): string {
  const statusClass = ep.status !== "pending" ? ` ${ep.status}` : "";
  const checked = ep.status === "pending" ? "checked" : "";
  const disabled = ep.status !== "pending" ? "disabled" : "";

  let progressHtml = "";
  if (ep.status === "downloading") {
    progressHtml = `<div class="ep-progress">
      <div class="ep-progress-bar"><div class="ep-progress-fill" id="feed-prog-${idx}" style="width:${Math.round(ep.progress * 100)}%"></div></div>
      <div class="ep-status">${ep.downloadedMb.toFixed(1)}/${ep.totalMb.toFixed(1)} MB</div>
    </div>`;
  } else if (ep.status === "complete") {
    progressHtml = `<div class="ep-progress"><div class="ep-status" style="color:var(--primary);">Done</div></div>`;
  } else if (ep.status === "error") {
    progressHtml = `<div class="ep-progress"><div class="ep-status">Failed</div></div>`;
  }

  return `<div class="feed-episode${statusClass}" data-ep-idx="${idx}">
    <input type="checkbox" ${checked} ${disabled} data-idx="${idx}" />
    <span class="ep-title">${escapeHtml(ep.title)}</span>
    <span class="ep-date">${ep.date}</span>
    ${progressHtml}
  </div>`;
}

function feedRenderList() {
  const feedEpisodes = document.getElementById("feed-episodes")!;
  feedEpisodes.innerHTML = feedEpisodesList.map((ep, i) => feedRenderEpisode(ep, i)).join("");
  // Wire checkbox change events
  feedEpisodes.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    cb.addEventListener("change", feedUpdateSelectedCount);
  });
  feedUpdateSelectedCount();
}

async function loadFeed(url: string) {
  const feedStatus = document.getElementById("feed-status")!;
  const feedActions = document.getElementById("feed-actions")!;

  feedStatus.textContent = "Loading feed...";
  feedActions.style.display = "none";
  feedEpisodesList = [];
  feedRenderList();

  try {
    const response = await fetch(url);
    const text = await response.text();
    const parser = new DOMParser();
    const xml = parser.parseFromString(text, "text/xml");
    const items = xml.querySelectorAll("item");

    if (items.length === 0) {
      feedStatus.textContent = "No episodes found in feed.";
      return;
    }

    feedEpisodesList = [];
    items.forEach((item, i) => {
      const title = item.querySelector("title")?.textContent || `Episode ${i + 1}`;
      const date = item.querySelector("pubDate")?.textContent || "";
      const enclosure = item.querySelector("enclosure");
      const audioUrl = enclosure?.getAttribute("url") || "";
      if (audioUrl) {
        feedEpisodesList.push({
          id: `ep-${i}`,
          title,
          date: date ? new Date(date).toLocaleDateString() : "",
          rawDate: date,
          audioUrl,
          status: "pending",
          progress: 0,
          downloadedMb: 0,
          totalMb: 0,
        });
      }
    });

    feedStatus.textContent = `Found ${feedEpisodesList.length} episodes. Select episodes to download:`;
    feedActions.style.display = "flex";
    feedRenderList();
  } catch (err) {
    feedStatus.textContent = `Failed to load feed: ${err}`;
  }
}

// Feed action buttons
document.getElementById("btn-feed-select-all")!.addEventListener("click", () => {
  document.getElementById("feed-episodes")!.querySelectorAll('input[type="checkbox"]:not(:disabled)').forEach(cb => {
    (cb as HTMLInputElement).checked = true;
  });
  feedUpdateSelectedCount();
});

document.getElementById("btn-feed-select-none")!.addEventListener("click", () => {
  document.getElementById("feed-episodes")!.querySelectorAll('input[type="checkbox"]:not(:disabled)').forEach(cb => {
    (cb as HTMLInputElement).checked = false;
  });
  feedUpdateSelectedCount();
});

document.getElementById("btn-feed-download")!.addEventListener("click", async () => {
  const feedEpisodes = document.getElementById("feed-episodes")!;
  const feedStatus = document.getElementById("feed-status")!;
  const checked = feedEpisodes.querySelectorAll('input[type="checkbox"]:checked:not(:disabled)') as NodeListOf<HTMLInputElement>;
  if (checked.length === 0) { alert("No episodes selected."); return; }

  // Ask for save directory
  const dir = await open({ directory: true, title: "Select download folder for episodes" });
  if (!dir) return;
  feedSaveDir = Array.isArray(dir) ? dir[0] : dir;

  const selectedIndices = Array.from(checked).map(cb => parseInt(cb.dataset.idx!));

  // Disable checkboxes and button
  (document.getElementById("btn-feed-download") as HTMLButtonElement).disabled = true;
  feedStatus.textContent = `Scanning for existing files...`;

  // Check which episodes already exist in the save dir
  let skipped = 0;
  let newlyTagged = 0;
  let alreadyTagged = 0;
  for (const idx of selectedIndices) {
    const ep = feedEpisodesList[idx];
    const safeName = ep.title.replace(/[^a-zA-Z0-9\s\-_.()]/g, "").replace(/\s+/g, "_").substring(0, 100);
    const urlPath = new URL(ep.audioUrl).pathname;
    const ext = urlPath.match(/\.(mp3|m4a|ogg|wav|flac|aac|opus)$/i)?.[0] || ".mp3";
    const outputPath = `${feedSaveDir}/${safeName}${ext}`;
    try {
      if (await invoke<boolean>("file_exists", { path: outputPath })) {
        const dateStr = ep.rawDate ? (() => { const d = new Date(ep.rawDate); return isNaN(d.getTime()) ? "" : `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`; })() : "";
        const result = await invoke<string>("download_podcast_episode", {
          url: ep.audioUrl, outputPath, episodeId: ep.id, comment: dateStr,
        });
        ep.status = "complete";
        ep.localPath = outputPath;
        ep.progress = 1;
        skipped++;
        if (result.startsWith("exists_tagged:")) newlyTagged++;
        else alreadyTagged++;
      }
    } catch {}
  }
  if (skipped > 0) feedRenderList();

  const remaining = selectedIndices.filter(i => feedEpisodesList[i].status === "pending");
  const tagInfo = newlyTagged > 0
    ? ` (${newlyTagged} newly dated, ${alreadyTagged} already had dates)`
    : alreadyTagged > 0 ? ` (all already had dates)` : "";

  if (remaining.length === 0) {
    feedStatus.textContent = `All ${skipped} episodes already exist${tagInfo}. Added to queue.`;
    (document.getElementById("btn-feed-download") as HTMLButtonElement).disabled = false;
    for (const idx of selectedIndices) {
      const ep = feedEpisodesList[idx];
      if (ep.localPath) addFiles([ep.localPath]);
    }
    return;
  }

  feedStatus.textContent = `${skipped} already exist${tagInfo}. Downloading ${remaining.length} remaining to ${feedSaveDir}...`;

  // Download concurrently (up to 3 at a time)
  const concurrency = 3;
  let completed = skipped;
  let errors = 0;
  const total = selectedIndices.length;
  const downloadQueue = [...remaining];

  async function downloadNext(): Promise<void> {
    while (downloadQueue.length > 0) {
      const idx = downloadQueue.shift()!;
      const ep = feedEpisodesList[idx];
      ep.status = "downloading";
      feedRenderList();

      // Derive filename from title
      const safeName = ep.title.replace(/[^a-zA-Z0-9\s\-_.()]/g, "").replace(/\s+/g, "_").substring(0, 100);
      // Get extension from URL
      const urlPath = new URL(ep.audioUrl).pathname;
      const ext = urlPath.match(/\.(mp3|m4a|ogg|wav|flac|aac|opus)$/i)?.[0] || ".mp3";
      const outputPath = `${feedSaveDir}/${safeName}${ext}`;

      try {
        const comment = ep.rawDate ? (() => { const d = new Date(ep.rawDate); return isNaN(d.getTime()) ? "" : `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`; })() : "";
        await invoke("download_podcast_episode", {
          url: ep.audioUrl,
          outputPath,
          episodeId: ep.id,
          comment,
        });
        ep.status = "complete";
        ep.localPath = outputPath;
        ep.progress = 1;
        completed++;

        // Add to transcription queue
        addFiles([outputPath]);
      } catch (err) {
        ep.status = "error";
        errors++;
        console.warn(`Failed to download ${ep.title}:`, err);
      }
      feedRenderList();
      feedStatus.textContent = `Downloaded ${completed}/${total}${errors > 0 ? ` (${errors} failed)` : ""}...`;
    }
  }

  const workers = Array.from({ length: Math.min(concurrency, selectedIndices.length) }, () => downloadNext());
  await Promise.all(workers);

  feedStatus.textContent = `Done. Downloaded ${completed}/${total} episodes${errors > 0 ? ` (${errors} failed)` : ""}.`;
  (document.getElementById("btn-feed-download") as HTMLButtonElement).disabled = false;
});

// Event listeners
btnAddFiles.addEventListener("click", async () => {
  const selected = await open({
    multiple: true,
    filters: [
      { name: "Audio & Video", extensions: ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus", "mp4", "m4v", "mov"] },
    ],
  });
  if (selected) {
    const paths = Array.isArray(selected) ? selected : [selected];
    await addFiles(paths);
  }
});

// Drag-and-drop of files/folders anywhere on the window. The webview
// intercepts native drags (Tauri dragDropEnabled default), so HTML5 drop
// events never fire — Tauri's own event is the only channel.
const DROP_EXTENSIONS = ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus", "mp4", "m4v", "mov"];

getCurrentWebview().onDragDropEvent(async (event) => {
  if (event.payload.type === "enter") {
    document.body.classList.add("drag-over");
  } else if (event.payload.type === "leave") {
    document.body.classList.remove("drag-over");
  } else if (event.payload.type === "drop") {
    document.body.classList.remove("drag-over");
    const files: string[] = [];
    for (const p of event.payload.paths) {
      try {
        // Directories expand to their supported contents (recursive)
        files.push(...await invoke<string[]>("scan_directory", { path: p }));
      } catch {
        const ext = p.split(".").pop()?.toLowerCase() || "";
        if (DROP_EXTENSIONS.includes(ext)) files.push(p);
      }
    }
    if (files.length > 0) await addFiles(files);
  }
});

btnAddFolder.addEventListener("click", async () => {
  const selected = await open({ directory: true });
  if (selected) {
    const dir = Array.isArray(selected) ? selected[0] : selected;
    const files = await invoke<string[]>("scan_directory", { path: dir });
    if (files.length === 0) alert("No audio files found in the selected directory.");
    else await addFiles(files);
  }
});

btnTranscribeAll.addEventListener("click", () => transcribeAll());

btnClearQueue.addEventListener("click", async () => {
  const active = queue.filter((q) => q.status === "transcribing" || q.status === "detecting" || q.status === "queued");
  if (active.length > 0) {
    renderPaused = true;
    const ok = confirm("There are active transcriptions. Clear anyway?");
    renderPaused = false;
    if (!ok) return;
  }
  // Cancel all active/queued jobs
  for (const item of active) {
    item.status = "cancelled";
    try { await invoke("cancel_job", { jobId: item.id }); } catch {}
  }
  queue.length = 0;
  renderQueue();
});

btnRetryErrors.addEventListener("click", async () => {
  const model = selectModel.value;
  const modelReady = await checkModelAndPromptDownload(model);
  if (!modelReady) return;

  const concurrent = parseInt(selectConcurrent.value);
  await invoke("set_concurrency", { permits: concurrent });

  const errorItems = queue.filter((q) => q.status === "error" || q.status === "cancelled");
  errorItems.forEach((item) => { item.status = "pending"; item.progress = 0; item.error = undefined; item.elapsed = undefined; });
  renderQueue();

  const promises = errorItems.map((item) => transcribeItem(item));
  await Promise.all(promises);
});

btnClearDone.addEventListener("click", () => {
  const doneIds = queue.filter((q) => q.status === "complete").map((q) => q.id);
  doneIds.forEach((id) => { const idx = queue.findIndex((q) => q.id === id); if (idx !== -1) queue.splice(idx, 1); });
  renderQueue();
});

selectOutput.addEventListener("change", async () => {
  if (selectOutput.value === "custom") {
    const selected = await open({ directory: true });
    if (selected) customOutputDir = Array.isArray(selected) ? selected[0] : selected;
    else { selectOutput.value = "same"; customOutputDir = null; }
  } else {
    customOutputDir = null;
  }
});

selectConcurrent.addEventListener("change", async () => {
  await invoke("set_concurrency", { permits: parseInt(selectConcurrent.value) });
});

// Modals
function setupModal(btnOpen: HTMLElement, modal: HTMLElement) {
  btnOpen.addEventListener("click", () => modal.classList.remove("hidden"));
  modal.querySelector(".modal-close")!.addEventListener("click", () => modal.classList.add("hidden"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); });
}

setupModal(btnModels, modelModal);
setupModal(btnSettings, settingsModal);
setupModal(btnAddFeed, feedModal);
setupModal(document.getElementById("btn-about")!, document.getElementById("about-modal")!);

// Speaker modal close + save handlers
const speakersModal = document.getElementById("speakers-modal")!;
speakersModal.querySelector(".modal-close")!.addEventListener("click", closeSpeakerModal);
speakersModal.addEventListener("click", (e) => { if (e.target === speakersModal) closeSpeakerModal(); });
document.getElementById("btn-speakers-cancel")!.addEventListener("click", closeSpeakerModal);
document.getElementById("btn-speakers-save")!.addEventListener("click", () => { saveSpeakerNames(); });

btnModels.addEventListener("click", () => refreshModelList());

// Settings
document.getElementById("btn-save-settings")!.addEventListener("click", () => {
  saveSettings();
  settingsModal.classList.add("hidden");
});

function loadSettingsIntoForm() {
  const s = loadSettings();
  (document.getElementById("input-api-key") as HTMLInputElement).value = s.apiKey;
  (document.getElementById("input-assemblyai-key") as HTMLInputElement).value = s.assemblyaiKey;
  (document.getElementById("select-assemblyai-model") as HTMLSelectElement).value = s.assemblyaiModel;
  (document.getElementById("input-assemblyai-lang") as HTMLInputElement).value = s.assemblyaiLang;
  (document.getElementById("chk-aai-trim-silence") as HTMLInputElement).checked = s.assemblyaiTrimSilence;
  (document.getElementById("input-aai-silence-db") as HTMLInputElement).value = String(s.assemblyaiSilenceDb);
  (document.getElementById("input-aai-min-silence") as HTMLInputElement).value = String(s.assemblyaiMinSilenceSecs);
  (document.getElementById("input-aai-silence-pad") as HTMLInputElement).value = String(s.assemblyaiSilencePadSecs);
  (document.getElementById("input-deepgram-key") as HTMLInputElement).value = s.deepgramKey;
  (document.getElementById("select-deepgram-model") as HTMLSelectElement).value = s.deepgramModel;
  (document.getElementById("input-deepgram-lang") as HTMLInputElement).value = s.deepgramLang;
  (document.getElementById("input-sherpa-threshold") as HTMLInputElement).value = String(s.sherpaThreshold);
  (document.getElementById("input-llm-models") as HTMLTextAreaElement).value = s.llmModels;
  (document.getElementById("input-api-url") as HTMLInputElement).value = s.apiUrl;
  if (s.chapterPrompt) (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement).value = s.chapterPrompt;
  (document.getElementById("select-chapter-format") as HTMLSelectElement).value = s.chapterOutputFormat;
  (document.getElementById("input-align-before") as HTMLInputElement).value = localStorage.getItem("align_before") || "2";
  (document.getElementById("input-align-after") as HTMLInputElement).value = localStorage.getItem("align_after") || "2";
  refreshLlmDropdown();
}

// Feed
document.getElementById("btn-load-feed")!.addEventListener("click", () => {
  const url = (document.getElementById("input-feed-url") as HTMLInputElement).value.trim();
  if (url) loadFeed(url);
});

// Rename speakers (standalone)
document.getElementById("btn-rename-speakers")!.addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Diarized transcripts", extensions: ["srt", "txt", "json"] }],
  });
  if (!selected) return;
  const path = Array.isArray(selected) ? selected[0] : selected;
  await openStandaloneSpeakerModal(path);
});

// Convert to CUE
document.getElementById("btn-convert-cue")!.addEventListener("click", async () => {
  const selected = await open({
    multiple: true,
    filters: [{ name: "Chapters", extensions: ["json", "srt", "txt"] }],
  });
  if (!selected) return;
  const paths = Array.isArray(selected) ? selected : [selected];
  let count = 0;
  for (const path of paths) {
    try {
      const text = await invoke<string>("read_text_file", { path });
      let chapters: Chapter[] = [];
      if (path.endsWith(".json")) {
        const data = JSON.parse(text);
        chapters = Array.isArray(data) ? data : (data.chapters || []);
      } else if (path.endsWith(".srt")) {
        // Parse SRT → chapter-like entries (each subtitle as a chapter)
        // More useful: look for chapter markers in the text
        const lines = text.split("\n");
        for (const line of lines) {
          const m = line.match(/^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*[-–]\s*(.+)/);
          if (m) chapters.push({ title: m[4], start_time: `${m[1]||"00"}:${m[2].padStart(2,"0")}:${String(Math.floor(+m[3])).padStart(2,"0")}`, start_secs: (+m[1]||0)*3600 + +m[2]*60 + +m[3] });
        }
        if (chapters.length === 0) {
          // fallback: parse as JSON chapters text format
          for (const line of lines) {
            const m2 = line.match(/(\d+:\d+:\d+)\s*[-–]\s*(.+)/);
            if (m2) {
              const parts = m2[1].split(":").map(Number);
              chapters.push({ title: m2[2], start_time: m2[1], start_secs: parts[0]*3600+parts[1]*60+parts[2] });
            }
          }
        }
      }
      if (chapters.length === 0) continue;
      // Find matching audio file (same stem, audio extension)
      const dir = path.substring(0, path.lastIndexOf("/"));
      const stem = path.split("/").pop()!.replace(/[._](chapters|llm_chapters|llm_raw|gemini[^.]*|gpt[^.]*|claude[^.]*)\..+$/, "").replace(/\.[^.]+$/, "");
      const exts = ["flac", "mp3", "wav", "m4a", "ogg", "aac"];
      let audioPath = "";
      for (const ext of exts) {
        const candidate = `${dir}/${stem}.${ext}`;
        const exists = await invoke<boolean>("file_exists", { path: candidate });
        if (exists) { audioPath = candidate; break; }
      }
      if (!audioPath) { console.warn(`No audio file found for ${stem}`); continue; }
      await invoke("write_cue_file", { audioPath, chapters });
      count++;
    } catch (e) { console.warn("CUE convert failed for", path, e); }
  }
  if (count > 0) alert(`Wrote ${count} .cue file(s)`);
});

// Parse a CUE file text into chapters
function parseCueText(text: string): Chapter[] {
  const chapters: Chapter[] = [];
  let currentTitle = "";
  for (const line of text.split("\n")) {
    const titleMatch = line.match(/^\s*TITLE\s+"(.+)"/);
    if (titleMatch) currentTitle = titleMatch[1];
    const indexMatch = line.match(/^\s*INDEX\s+01\s+(\d+):(\d+):(\d+)/);
    if (indexMatch) {
      const totalSecs = parseInt(indexMatch[1]) * 60 + parseInt(indexMatch[2]) + parseInt(indexMatch[3]) / 75;
      const h = Math.floor(totalSecs / 3600);
      const m = Math.floor((totalSecs % 3600) / 60);
      const s = Math.floor(totalSecs % 60);
      chapters.push({
        title: currentTitle || `Chapter ${chapters.length + 1}`,
        start_time: `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`,
        start_secs: totalSecs,
      });
      currentTitle = "";
    }
  }
  return chapters;
}

// Find audio file for a CUE file
async function findAudioForCue(cuePath: string, cueText: string): Promise<string | null> {
  const dir = cuePath.substring(0, cuePath.lastIndexOf("/"));
  const fileMatch = cueText.match(/^\s*FILE\s+"(.+?)"/m);
  if (fileMatch) {
    const candidate = `${dir}/${fileMatch[1]}`;
    if (await invoke<boolean>("file_exists", { path: candidate })) return candidate;
  }
  const stem = cuePath.split("/").pop()!.replace(/\.cue$/, "");
  for (const ext of ["flac", "mp3", "wav", "m4a", "ogg", "aac"]) {
    const candidate = `${dir}/${stem}.${ext}`;
    if (await invoke<boolean>("file_exists", { path: candidate })) return candidate;
  }
  return null;
}

// Try to load pipeline markers (raw LLM, SRT corrected) for an audio file
async function loadPipelineMarkers(audioPath: string, chapters: Chapter[]): Promise<ChapterWithSnap[] | undefined> {
  const dir = audioPath.substring(0, audioPath.lastIndexOf("/"));
  const stem = audioPath.split("/").pop()!.replace(/\.[^.]+$/, "");
  const settings = loadSettings();
  const llmShort = llmModelShort(settings.llmModel);

  let rawChapters: Chapter[] | null = null;
  let correctedChapters: Chapter[] | null = null;

  // Try to load raw LLM chapters
  const rawPath = `${dir}/${stem}_${llmShort}_llm_raw.json`;
  try {
    const data = await invoke<string>("read_text_file", { path: rawPath });
    rawChapters = JSON.parse(data);
  } catch {}

  // Try to load SRT-corrected chapters
  const corrPath = `${dir}/${stem}_${llmShort}_llm_chapters.json`;
  try {
    const data = await invoke<string>("read_text_file", { path: corrPath });
    correctedChapters = JSON.parse(data);
  } catch {}

  // Try to load YAMNet onsets
  let yamnetOnsets: number[] | null = null;
  const onsetsPath = `${dir}/${stem}_yamnet_onsets.json`;
  try {
    const data = await invoke<string>("read_text_file", { path: onsetsPath });
    yamnetOnsets = JSON.parse(data);
  } catch {}

  if (!rawChapters && !correctedChapters && !yamnetOnsets) return undefined;

  return chapters.map((ch, i) => {
    const rawSecs = rawChapters?.[i]?.start_secs ?? ch.start_secs;
    const corrSecs = correctedChapters?.[i]?.start_secs ?? null;
    // Find nearest YAMNet onset within ±60s
    let yamSecs: number | null = null;
    if (yamnetOnsets) {
      let best: number | null = null;
      let bestDist = 60;
      for (const o of yamnetOnsets) {
        const d = Math.abs(o - ch.start_secs);
        if (d < bestDist) { bestDist = d; best = o; }
      }
      yamSecs = best;
    }
    return {
      title: ch.title,
      start_time: ch.start_time,
      start_secs: ch.start_secs,
      raw_llm_secs: chkFirstZero.checked && i === 0 ? 0 : rawSecs,
      srt_corrected_secs: corrSecs,
      yamnet_secs: yamSecs,
      snapped: false,
    };
  });
}

// Align CUE — load CUE file(s), show list with align buttons
document.getElementById("btn-align-cue")!.addEventListener("click", async () => {
  const selected = await open({
    multiple: true,
    filters: [{ name: "CUE files", extensions: ["cue"] }],
  });
  if (!selected) return;
  const cuePaths = Array.isArray(selected) ? selected : [selected];

  if (cuePaths.length === 1) {
    // Single file — open align modal directly
    try {
      const text = await invoke<string>("read_text_file", { path: cuePaths[0] });
      const chapters = parseCueText(text);
      if (chapters.length === 0) { alert("No chapters found in CUE file"); return; }
      const audioPath = await findAudioForCue(cuePaths[0], text);
      if (!audioPath) { alert("No matching audio file found for this CUE file"); return; }
      const snap = await loadPipelineMarkers(audioPath, chapters);
      openAlignModal(audioPath, chapters, 0, snap);
    } catch (e) { alert(`Failed to load CUE: ${e}`); }
    return;
  }

  // Multiple files — show list in chapter-detail-modal
  const modal = document.getElementById("chapter-detail-modal")!;
  const title = document.getElementById("chapter-detail-title")!;
  const list = document.getElementById("chapter-detail-list")!;
  title.textContent = `Align CUE Files (${cuePaths.length})`;

  interface CueEntry { cuePath: string; audioPath: string; chapters: Chapter[]; snap?: ChapterWithSnap[]; name: string; }
  const entries: CueEntry[] = [];

  list.innerHTML = `<div style="color:var(--text-muted);padding:8px;">Loading ${cuePaths.length} CUE files...</div>`;
  modal.classList.remove("hidden");
  modal.querySelector(".modal-close")!.addEventListener("click", () => modal.classList.add("hidden"), { once: true });
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.add("hidden"); }, { once: true });

  for (const cp of cuePaths) {
    const name = cp.split("/").pop() || cp;
    try {
      const text = await invoke<string>("read_text_file", { path: cp });
      const chapters = parseCueText(text);
      const audioPath = await findAudioForCue(cp, text);
      if (audioPath && chapters.length > 0) {
        const snap = await loadPipelineMarkers(audioPath, chapters);
        entries.push({ cuePath: cp, audioPath, chapters, snap, name });
      } else {
        entries.push({ cuePath: cp, audioPath: "", chapters, name });
      }
    } catch {
      entries.push({ cuePath: cp, audioPath: "", chapters: [], name });
    }
  }

  let html = `<table class="chapter-compare-table"><thead><tr><th>#</th><th>File</th><th>Chapters</th><th></th></tr></thead><tbody>`;
  entries.forEach((entry, idx) => {
    const hasAudio = !!entry.audioPath;
    html += `<tr>
      <td>${idx + 1}</td>
      <td class="ch-title-cell">${escapeHtml(entry.name)}</td>
      <td>${entry.chapters.length}</td>
      <td>${hasAudio
        ? `<button class="small cue-align-btn" data-cue-idx="${idx}">Align</button>`
        : `<span style="color:var(--text-muted)">No audio</span>`}</td>
    </tr>`;
  });
  html += `</tbody></table>`;
  list.innerHTML = html;

  list.querySelectorAll(".cue-align-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = +(btn as HTMLElement).dataset.cueIdx!;
      const entry = entries[idx];
      modal.classList.add("hidden");
      openAlignModal(entry.audioPath, entry.chapters, 0, entry.snap);
    });
  });
});

// Chapter detection standalone
const chaptersModal = document.getElementById("chapters-modal")!;
const btnDetectChapters = document.getElementById("btn-detect-chapters")!;
const btnChooseChapterFile = document.getElementById("btn-choose-chapter-file")!;
const inputChapterFile = document.getElementById("input-chapter-file") as HTMLInputElement;
const btnRunDetect = document.getElementById("btn-run-detect")! as HTMLButtonElement;
const chaptersStatus = document.getElementById("chapters-status")!;
const chaptersResults = document.getElementById("chapters-results")!;

let chapterFilePaths: string[] = [];

setupModal(btnDetectChapters, chaptersModal);
// Sync modal checkboxes with settings bar
btnDetectChapters.addEventListener("click", () => {
  (document.getElementById("chk-modal-snap") as HTMLInputElement).checked = chkSnapGaps.checked;
  (document.getElementById("chk-modal-embed") as HTMLInputElement).checked = chkEmbedFlac.checked;
  (document.getElementById("chk-modal-cue") as HTMLInputElement).checked = chkWriteCue.checked;
});

btnChooseChapterFile.addEventListener("click", async () => {
  const selected = await open({
    multiple: true,
    filters: [{ name: "Transcript", extensions: ["srt", "vtt", "txt", "json"] }],
  });
  if (selected) {
    chapterFilePaths = Array.isArray(selected) ? selected : [selected];
    inputChapterFile.value = chapterFilePaths.length === 1
      ? chapterFilePaths[0]
      : `${chapterFilePaths.length} files selected`;
  }
});

btnRunDetect.addEventListener("click", async () => {
  if (chapterFilePaths.length === 0) {
    chaptersStatus.textContent = "Please select transcript file(s) first.";
    chaptersStatus.className = "chapters-status error";
    return;
  }

  const settings = loadSettings();
  if (!settings.apiKey) {
    chaptersStatus.textContent = "No API key configured. Go to Settings first.";
    chaptersStatus.className = "chapters-status error";
    return;
  }

  const chkModalSnap = document.getElementById("chk-modal-snap") as HTMLInputElement;
  const useSnap = chkModalSnap.checked;

  btnRunDetect.disabled = true;
  chaptersResults.innerHTML = "";

  for (let i = 0; i < chapterFilePaths.length; i++) {
    const filePath = chapterFilePaths[i];
    const fileName = filePath.split("/").pop() || filePath;

    chaptersStatus.textContent = chapterFilePaths.length > 1
      ? `Processing ${i + 1}/${chapterFilePaths.length}: ${fileName}...`
      : `Reading and analyzing ${fileName}...`;
    chaptersStatus.className = "chapters-status";

    try {
      const transcript = await invoke<string>("read_text_file", { path: filePath });

      let chapters: Chapter[];
      let snappedChaptersResult: ChapterWithSnap[] | null = null;

      if (useSnap) {
        // Try to find the source audio file alongside the transcript
        const dir = filePath.substring(0, filePath.lastIndexOf("/"));
        // Strip _transcription_<model> suffix to get the audio stem
        const rawStem = fileName.replace(/\.[^.]+$/, "");
        const stem = rawStem.replace(/_transcription_[^_]+$/, "") || rawStem;
        const audioExts = ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus"];
        let audioPath = "";
        for (const ext of audioExts) {
          const candidate = `${dir}/${stem}.${ext}`;
          const exists = await invoke<boolean>("file_exists", { path: candidate });
          if (exists) { audioPath = candidate; break; }
        }

        if (!audioPath) {
          // Fall back to LLM-only if no audio file found
          chaptersResults.innerHTML += `<div class="chapter-file-header">${escapeHtml(fileName)} - No matching audio file found, using LLM only</div>`;
          chapters = await invoke<Chapter[]>("detect_chapters", {
            req: { transcript, api_key: settings.apiKey, model: settings.llmModel, base_url: settings.apiUrl, prompt: settings.chapterPrompt, transcript_path: filePath, raw_mode: !chkSrtCorrect.checked },
          });
        } else {
          chaptersStatus.textContent = chapterFilePaths.length > 1
            ? `Processing ${i + 1}/${chapterFilePaths.length}: ${fileName} (LLM + YAMNet snap)...`
            : `Analyzing ${fileName} with LLM + YAMNet snap...`;

          snappedChaptersResult = await invoke<ChapterWithSnap[]>("detect_chapters_with_gaps", {
            req: { transcript, api_key: settings.apiKey, model: settings.llmModel, base_url: settings.apiUrl, prompt: settings.chapterPrompt, transcript_path: filePath, raw_mode: !chkSrtCorrect.checked },
            audioPath,
            minGapSecs: 1.5,
            silenceThreshold: 0.02,
            maxLookbackSecs: 60.0,
          });
          chapters = snappedChaptersResult.map((ch) => ({
            title: ch.title,
            start_time: ch.start_time,
            start_secs: ch.start_secs,
          }));
        }
      } else {
        chapters = await invoke<Chapter[]>("detect_chapters", {
          req: { transcript, api_key: settings.apiKey, model: settings.llmModel, base_url: settings.apiUrl, prompt: settings.chapterPrompt, transcript_path: filePath, raw_mode: !chkSrtCorrect.checked },
        });
      }

      // Force first chapter to 0:00 if checkbox is checked
      if (chkFirstZero.checked && chapters.length > 0 && chapters[0].start_secs > 0) {
        chapters[0].start_secs = 0;
        chapters[0].start_time = "00:00:00";
      }

      if (chapters.length === 0) {
        chaptersResults.innerHTML += `<div class="chapter-file-header">${escapeHtml(fileName)} - No chapters detected</div>`;
        continue;
      }

      // Auto-save
      const dir = filePath.substring(0, filePath.lastIndexOf("/"));
      const rawStemSave = fileName.replace(/\.[^.]+$/, "");
      const stem = rawStemSave.replace(/_transcription_[^_]+$/, "") || rawStemSave;
      const chapterFormat = settings.chapterOutputFormat || "txt";

      let content: string;
      let ext: string;
      if (chapterFormat === "json") {
        content = JSON.stringify(chapters, null, 2);
        ext = "json";
      } else {
        content = chapters.map((ch) => `${ch.start_time} - ${ch.title}`).join("\n");
        ext = "txt";
      }

      const outPath = `${customOutputDir || dir}/${stem}.${ext}`;
      await invoke("write_text_file", { path: outPath, content });

      // Embed chapters in FLAC and write .cue if audio file found
      const chkModalEmbed = document.getElementById("chk-modal-embed") as HTMLInputElement;
      const chkModalCue = document.getElementById("chk-modal-cue") as HTMLInputElement;
      if (chkModalEmbed.checked || chkModalCue.checked) {
        const audioExtsEmbed = ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus"];
        for (const aext of audioExtsEmbed) {
          const candidate = `${dir}/${stem}.${aext}`;
          const exists = await invoke<boolean>("file_exists", { path: candidate });
          if (exists) {
            if (chkModalEmbed.checked) {
              try { await invoke("embed_chapters_in_flac", { req: { audio_path: candidate, chapters } }); } catch (e) { console.warn("Embed failed:", e); }
            }
            if (chkModalCue.checked) {
              try { await invoke("write_cue_file", { audioPath: candidate, chapters }); } catch (e) { console.warn("Cue failed:", e); }
            }
            break;
          }
        }
      }

      // Display — highlight snapped chapters
      let html = `<div class="chapter-file-header">${escapeHtml(fileName)} (${chapters.length} chapters) <span class="saved-badge">Saved: ${stem}.${ext}</span></div>`;
      html += chapters.map((ch, idx) => {
        const wasSnapped = snappedChaptersResult && snappedChaptersResult[idx]?.snapped;
        return `
        <div class="chapter-item${wasSnapped ? " snapped" : ""}">
          <span class="ch-time">${escapeHtml(ch.start_time)}</span>
          <span class="ch-title">${escapeHtml(ch.title)}${wasSnapped ? ' <span class="snapped-badge">snapped</span>' : ""}</span>
        </div>`;
      }).join("");
      chaptersResults.innerHTML += html;
    } catch (err: any) {
      const errMsg = typeof err === "string" ? err : err?.message || "Failed";
      chaptersResults.innerHTML += `<div class="chapter-file-header error">${escapeHtml(fileName)} - Error: ${escapeHtml(errMsg)}</div>`;
    }
  }

  chaptersStatus.textContent = `Done processing ${chapterFilePaths.length} file(s).`;
  btnRunDetect.disabled = false;
});

// Events
listen<TranscriptionProgress>("transcription-progress", (event) => {
  const data = event.payload;
  const item = queue.find((q) => q.id === data.job_id);
  if (item) {
    item.progress = data.progress;
    item.stageText = formatStageText(data.status);
    const isActiveStage =
      data.status === "transcribing" ||
      data.status === "uploading" ||
      data.status === "submitting" ||
      data.status === "diarizing" ||
      data.status === "merging" ||
      data.status === "loading_model" ||
      data.status.startsWith("processing");
    if (isActiveStage && item.status !== "transcribing") {
      item.status = "transcribing";
      item.startedAt = Date.now();
    }
    renderQueue();
  }
});

function formatStageText(status: string): string {
  if (status === "uploading") return "Uploading…";
  if (status === "submitting") return "Submitting…";
  if (status === "processing") return "Processing…";
  if (status.startsWith("processing (")) {
    const inner = status.substring("processing (".length, status.length - 1);
    return `Processing (${inner})…`;
  }
  if (status === "loading_model") return "Loading model…";
  if (status === "transcribing") return "Transcribing…";
  if (status === "diarizing") return "Diarizing…";
  if (status === "merging") return "Merging speakers…";
  if (status === "complete") return "Complete";
  return status;
}

listen<ModelDownloadProgress>("model-download-progress", (event) => {
  const data = event.payload;
  const fill = document.querySelector(`.model-dl-fill[data-model="${data.model}"]`) as HTMLElement;
  const text = document.querySelector(`.model-dl-text[data-model="${data.model}"]`) as HTMLElement;
  if (fill) fill.style.width = `${Math.round(data.progress * 100)}%`;
  if (text && data.downloaded_mb !== undefined) text.textContent = `${data.downloaded_mb.toFixed(0)} / ${data.total_mb?.toFixed(0)} MB`;
});

listen<{status: string; progress: number}>("gap-progress", (event) => {
  const data = event.payload;
  // Update the chapters status if modal is open
  const chaptersStatus = document.getElementById("chapters-status");
  if (chaptersStatus && !chaptersModal.classList.contains("hidden")) {
    chaptersStatus.textContent = data.status;
    chaptersStatus.className = "chapters-status";
  }
  // Also update any "detecting" queue items
  const detectingItem = queue.find(q => q.status === "detecting");
  if (detectingItem) {
    const statusEl = document.querySelector(`[data-id="${detectingItem.id}"] .status, #queue-list .status`);
    // Just re-render queue item to show progress would be heavy; store for display
    detectingItem.detectStatus = data.status;
    const el = document.getElementById(`detect-status-${detectingItem.id}`);
    if (el) el.textContent = data.status;
  }
});

listen<{episode_id: string; progress: number; status: string; downloaded_mb?: number; total_mb?: number}>("podcast-download-progress", (event) => {
  const data = event.payload;
  const ep = feedEpisodesList.find(e => e.id === data.episode_id);
  if (ep) {
    ep.progress = data.progress;
    ep.downloadedMb = data.downloaded_mb || 0;
    ep.totalMb = data.total_mb || 0;
    // Update progress bar in-place without full re-render
    const idx = feedEpisodesList.indexOf(ep);
    const fill = document.getElementById(`feed-prog-${idx}`);
    if (fill) {
      fill.style.width = `${Math.round(data.progress * 100)}%`;
      const statusEl = fill.closest(".ep-progress")?.querySelector(".ep-status");
      if (statusEl) statusEl.textContent = `${ep.downloadedMb.toFixed(1)}/${ep.totalMb.toFixed(1)} MB`;
    }
  }
});

// Persist UI preferences
function savePreferences() {
  localStorage.setItem("pref_engine", selectEngine.value);
  localStorage.setItem("pref_model", selectModel.value);
  localStorage.setItem("pref_format", selectFormat.value);
  localStorage.setItem("pref_threads", selectThreads.value);
  localStorage.setItem("pref_concurrent", selectConcurrent.value);
  localStorage.setItem("pref_auto_chapters", chkAutoChapters.checked ? "1" : "0");
  localStorage.setItem("pref_snap_gaps", chkSnapGaps.checked ? "1" : "0");
  localStorage.setItem("pref_embed_flac", chkEmbedFlac.checked ? "1" : "0");
  localStorage.setItem("pref_write_cue", chkWriteCue.checked ? "1" : "0");
  localStorage.setItem("pref_srt_correct", chkSrtCorrect.checked ? "1" : "0");
  localStorage.setItem("pref_per_word", chkPerWord.checked ? "1" : "0");
  localStorage.setItem("pref_first_zero", chkFirstZero.checked ? "1" : "0");
  localStorage.setItem("speakers_expected", inputSpeakersExpected.value.trim());
}

function loadPreferences() {
  const engine = localStorage.getItem("pref_engine");
  if (engine) selectEngine.value = engine;
  const model = localStorage.getItem("pref_model");
  if (model) selectModel.value = model;
  const format = localStorage.getItem("pref_format");
  if (format) selectFormat.value = format;
  const threads = localStorage.getItem("pref_threads");
  if (threads) selectThreads.value = threads;
  const concurrent = localStorage.getItem("pref_concurrent");
  if (concurrent) selectConcurrent.value = concurrent;
  const autoChapters = localStorage.getItem("pref_auto_chapters");
  if (autoChapters !== null) chkAutoChapters.checked = autoChapters === "1";
  const useVad = localStorage.getItem("pref_snap_gaps");
  if (useVad !== null) chkSnapGaps.checked = useVad === "1";
  const embedFlac = localStorage.getItem("pref_embed_flac");
  if (embedFlac !== null) chkEmbedFlac.checked = embedFlac === "1";
  const writeCue = localStorage.getItem("pref_write_cue");
  if (writeCue !== null) chkWriteCue.checked = writeCue === "1";
  const srtCorrect = localStorage.getItem("pref_srt_correct");
  if (srtCorrect !== null) chkSrtCorrect.checked = srtCorrect === "1";
  const perWord = localStorage.getItem("pref_per_word");
  if (perWord !== null) chkPerWord.checked = perWord === "1";
  const firstZero = localStorage.getItem("pref_first_zero");
  if (firstZero !== null) chkFirstZero.checked = firstZero === "1";
  const speakers = localStorage.getItem("speakers_expected") ?? localStorage.getItem("sherpa_num_speakers");
  if (speakers !== null && speakers !== "0") inputSpeakersExpected.value = speakers;

  // One-time migration: the Sherpa embedding moved from NeMo SpeakerNet to
  // WeSpeaker CAM++, which lives in a different distance space. Clear any saved
  // SpeakerNet-era threshold so the new CAM++ default (0.5) takes effect. Users
  // who deliberately tuned it afterward keep their value (flag set below).
  if (localStorage.getItem("sherpa_embed_ver") !== "campp_lm") {
    localStorage.removeItem("sherpa_threshold");
    localStorage.setItem("sherpa_embed_ver", "campp_lm");
  }
}

function applyEngineUI() {
  const bar = document.querySelector(".settings-bar");
  if (!bar) return;
  bar.classList.remove("engine-assemblyai", "engine-deepgram", "engine-sherpa", "engine-parakeet", "engine-parakeet-sherpa", "engine-compare-local", "engine-cloud");
  if (selectEngine.value === "assemblyai") {
    bar.classList.add("engine-assemblyai", "engine-cloud");
  } else if (selectEngine.value === "deepgram") {
    bar.classList.add("engine-deepgram", "engine-cloud");
  } else if (selectEngine.value === "sherpa") {
    bar.classList.add("engine-sherpa");
  } else if (selectEngine.value === "parakeet") {
    bar.classList.add("engine-parakeet");
  } else if (selectEngine.value === "parakeet-sherpa") {
    bar.classList.add("engine-parakeet-sherpa");
  } else if (selectEngine.value === "compare-local") {
    bar.classList.add("engine-compare-local");
  }
  bar.classList.toggle("no-auto-chapters", !chkAutoChapters.checked);
}
selectEngine.addEventListener("change", () => {
  savePreferences();
  applyEngineUI();
});
applyEngineUI();
selectModel.addEventListener("change", savePreferences);
selectFormat.addEventListener("change", savePreferences);
selectThreads.addEventListener("change", savePreferences);
selectConcurrent.addEventListener("change", savePreferences);
inputSpeakersExpected.addEventListener("change", savePreferences);
chkAutoChapters.addEventListener("change", () => {
  savePreferences();
  applyEngineUI();
});
chkSnapGaps.addEventListener("change", savePreferences);
chkEmbedFlac.addEventListener("change", savePreferences);
chkWriteCue.addEventListener("change", savePreferences);
chkSrtCorrect.addEventListener("change", savePreferences);
chkPerWord.addEventListener("change", savePreferences);
chkFirstZero.addEventListener("change", savePreferences);
selectLlmModel.addEventListener("change", () => {
  localStorage.setItem("llm_model", selectLlmModel.value);
});

// === Template Editor ===
interface WaveformData {
  peaks_min: number[];
  peaks_max: number[];
  start_secs: number;
  end_secs: number;
  duration_secs: number;
}

interface TemplateInfo {
  name: string;
  duration_secs: number;
  source_file: string;
}

interface TemplateMatchResult {
  time_secs: number;
  confidence: number;
}

const templateModal = document.getElementById("template-modal")!;
const btnTemplateEditor = document.getElementById("btn-template-editor")!;
const templateAudioPath = document.getElementById("template-audio-path") as HTMLInputElement;
const btnTemplateBrowse = document.getElementById("btn-template-browse")!;
const btnTemplateLoadChapters = document.getElementById("btn-template-load-chapters")!;
const templateJumpTime = document.getElementById("template-jump-time") as HTMLInputElement;
const btnTemplateJump = document.getElementById("btn-template-jump")!;
const templateCanvas = document.getElementById("template-waveform") as HTMLCanvasElement;
const templateTimeInfo = document.getElementById("template-time-info")!;
const btnTemplateZoomIn = document.getElementById("btn-template-zoom-in")!;
const btnTemplateZoomOut = document.getElementById("btn-template-zoom-out")!;
const btnTemplateZoomFit = document.getElementById("btn-template-zoom-fit")!;
const btnTemplateZoomSel = document.getElementById("btn-template-zoom-sel")!;
const btnTemplatePlay = document.getElementById("btn-template-play")!;
const btnTemplateSave = document.getElementById("btn-template-save")!;
const templateNameInput = document.getElementById("template-name-input") as HTMLInputElement;
const templateListItems = document.getElementById("template-list-items")!;
const templateThreshold = document.getElementById("template-threshold") as HTMLInputElement;
const templateThresholdVal = document.getElementById("template-threshold-val")!;
templateThreshold.addEventListener("input", () => { templateThresholdVal.textContent = templateThreshold.value; });

setupModal(btnTemplateEditor, templateModal);

let tmplAudioFile = "";
let tmplDuration = 0;
let tmplViewStart = 0;
let tmplViewEnd = 30; // show 30s initially
let tmplSelStart = 0;
let tmplSelEnd = 0;
let tmplDragging: "start" | "end" | "pan" | null = null;
let tmplDragStartX = 0;
let tmplPanOrigin = { viewStart: 0, viewEnd: 0 };
let tmplWaveform: WaveformData | null = null;
let tmplAudioCtx: AudioContext | null = null;
let tmplPlayingSource: AudioBufferSourceNode | null = null;
let tmplCssW = 800;
let tmplCssH = 180;
let tmplChapterMarkers: { time: number; title: string }[] = [];
let tmplPlayStartCtx = 0;
let tmplPlayStartSec = 0;
let tmplPlayheadRAF = 0;

function tmplSetupCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const rect = templateCanvas.getBoundingClientRect();
  tmplCssW = rect.width || 800;
  tmplCssH = rect.height || 180;
  templateCanvas.width = Math.round(tmplCssW * dpr);
  templateCanvas.height = Math.round(tmplCssH * dpr);
  const ctx = templateCanvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function tmplSecsToX(secs: number): number {
  return ((secs - tmplViewStart) / (tmplViewEnd - tmplViewStart)) * tmplCssW;
}

function tmplXToSecs(x: number): number {
  return tmplViewStart + (x / tmplCssW) * (tmplViewEnd - tmplViewStart);
}

function tmplFormatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(2);
  return `${m}:${s.padStart(5, "0")}`;
}

async function tmplLoadWaveform() {
  if (!tmplAudioFile) return;
  templateTimeInfo.textContent = "Loading waveform...";
  try {
    tmplWaveform = await invoke<WaveformData>("get_waveform_peaks", {
      path: tmplAudioFile,
      startSecs: tmplViewStart,
      endSecs: tmplViewEnd,
      numPoints: Math.round(tmplCssW),
    });
    tmplDrawWaveform();
    tmplUpdateInfo();
  } catch (e) {
    templateTimeInfo.textContent = `Error: ${e}`;
  }
}

let tmplLoadTimer = 0;
function tmplScheduleLoad() {
  clearTimeout(tmplLoadTimer);
  tmplLoadTimer = window.setTimeout(() => tmplLoadWaveform(), 80);
}

function tmplAnimatePlayhead() {
  if (!tmplPlayingSource) { tmplPlayheadRAF = 0; return; }
  tmplDrawWaveform();
  tmplPlayheadRAF = requestAnimationFrame(tmplAnimatePlayhead);
}

function tmplDrawWaveform() {
  const ctx = templateCanvas.getContext("2d")!;
  const w = tmplCssW;
  const h = tmplCssH;
  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = "#0a0a1a";
  ctx.fillRect(0, 0, w, h);

  if (!tmplWaveform) return;

  // Selection highlight
  const selX1 = tmplSecsToX(tmplSelStart);
  const selX2 = tmplSecsToX(tmplSelEnd);
  if (tmplSelEnd > tmplSelStart) {
    ctx.fillStyle = "rgba(0, 176, 240, 0.15)";
    ctx.fillRect(selX1, 0, selX2 - selX1, h);
  }

  // Waveform
  const mid = h / 2;
  const scale = mid * 0.9;
  ctx.strokeStyle = "#00b0f0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  const stepX = w / tmplWaveform.peaks_max.length;
  for (let i = 0; i < tmplWaveform.peaks_max.length; i++) {
    const x = i * stepX;
    const yMax = mid - tmplWaveform.peaks_max[i] * scale;
    const yMin = mid - tmplWaveform.peaks_min[i] * scale;
    ctx.moveTo(x, yMax);
    ctx.lineTo(x, yMin);
  }
  ctx.stroke();

  // Center line
  ctx.strokeStyle = "rgba(255,255,255,0.1)";
  ctx.beginPath();
  ctx.moveTo(0, mid);
  ctx.lineTo(w, mid);
  ctx.stroke();

  // Selection handles
  if (tmplSelEnd > tmplSelStart) {
    for (const x of [selX1, selX2]) {
      ctx.strokeStyle = "#ff9800";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();

      // Handle grip
      ctx.fillStyle = "#ff9800";
      ctx.fillRect(x - 4, 0, 8, 12);
      ctx.fillRect(x - 4, h - 12, 8, 12);
    }
  }

  // Time labels
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = "11px monospace";
  const viewDur = tmplViewEnd - tmplViewStart;
  const labelStep = viewDur < 5 ? 0.5 : viewDur < 30 ? 2 : viewDur < 120 ? 10 : 30;
  let t = Math.ceil(tmplViewStart / labelStep) * labelStep;
  while (t < tmplViewEnd) {
    const x = tmplSecsToX(t);
    ctx.fillText(tmplFormatTime(t), x + 2, h - 2);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    t += labelStep;
  }

  // Chapter markers overlay
  for (const m of tmplChapterMarkers) {
    if (m.time < tmplViewStart || m.time > tmplViewEnd) continue;
    const x = tmplSecsToX(m.time);
    ctx.strokeStyle = "rgba(76, 175, 80, 0.8)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(76, 175, 80, 0.9)";
    ctx.font = "10px sans-serif";
    ctx.fillText(m.title.slice(0, 30), x + 3, 11);
  }

  // Playhead
  if (tmplPlayingSource && tmplAudioCtx) {
    const playPos = tmplPlayStartSec + (tmplAudioCtx.currentTime - tmplPlayStartCtx);
    if (playPos >= tmplViewStart && playPos <= tmplViewEnd) {
      const x = tmplSecsToX(playPos);
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }
  }
}

function tmplUpdateInfo() {
  if (tmplSelEnd > tmplSelStart) {
    templateTimeInfo.textContent =
      `Selection: ${tmplFormatTime(tmplSelStart)} — ${tmplFormatTime(tmplSelEnd)} (${(tmplSelEnd - tmplSelStart).toFixed(2)}s)  |  View: ${tmplFormatTime(tmplViewStart)} — ${tmplFormatTime(tmplViewEnd)}`;
  } else {
    templateTimeInfo.textContent = `View: ${tmplFormatTime(tmplViewStart)} — ${tmplFormatTime(tmplViewEnd)}  |  Click and drag to select region`;
  }
}

// Canvas mouse interaction
templateCanvas.addEventListener("mousedown", (e) => {
  const rect = templateCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const secs = tmplXToSecs(x);

  // Check if clicking near a handle
  if (tmplSelEnd > tmplSelStart) {
    const startX = tmplSecsToX(tmplSelStart);
    const endX = tmplSecsToX(tmplSelEnd);
    if (Math.abs(x - startX) < 8) { tmplDragging = "start"; return; }
    if (Math.abs(x - endX) < 8) { tmplDragging = "end"; return; }
  }

  // Middle click or right-click to pan
  if (e.button === 1 || e.button === 2) {
    tmplDragging = "pan";
    tmplDragStartX = x;
    tmplPanOrigin = { viewStart: tmplViewStart, viewEnd: tmplViewEnd };
    return;
  }

  // Start new selection
  tmplSelStart = secs;
  tmplSelEnd = secs;
  tmplDragging = "end";
});

templateCanvas.addEventListener("mousemove", (e) => {
  if (!tmplDragging) return;
  const rect = templateCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const secs = tmplXToSecs(x);

  if (tmplDragging === "start") {
    tmplSelStart = Math.max(tmplViewStart, Math.min(secs, tmplSelEnd - 0.05));
  } else if (tmplDragging === "end") {
    tmplSelEnd = Math.min(tmplViewEnd, Math.max(secs, tmplSelStart + 0.05));
  } else if (tmplDragging === "pan") {
    const dx = (x - tmplDragStartX) / tmplCssW * (tmplPanOrigin.viewEnd - tmplPanOrigin.viewStart);
    const dur = tmplPanOrigin.viewEnd - tmplPanOrigin.viewStart;
    tmplViewStart = Math.max(0, Math.min(tmplDuration - dur, tmplPanOrigin.viewStart - dx));
    tmplViewEnd = tmplViewStart + dur;
    tmplScheduleLoad();
    tmplDrawWaveform();
    return;
  }
  tmplDrawWaveform();
  tmplUpdateInfo();
});

window.addEventListener("mouseup", () => {
  tmplDragging = null;
});

templateCanvas.addEventListener("contextmenu", (e) => e.preventDefault());

// Scroll to zoom
templateCanvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const rect = templateCanvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const center = tmplXToSecs(x);
  const viewDur = tmplViewEnd - tmplViewStart;
  const factor = e.deltaY > 0 ? 1.3 : 0.7;
  const newDur = Math.max(0.5, Math.min(tmplDuration, viewDur * factor));
  const ratio = (center - tmplViewStart) / viewDur;
  tmplViewStart = Math.max(0, center - newDur * ratio);
  tmplViewEnd = Math.min(tmplDuration, tmplViewStart + newDur);
  tmplLoadWaveform();
});

// Browse for audio file
btnTemplateBrowse.addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Audio", extensions: ["flac", "mp3", "wav", "ogg", "m4a", "aac"] }],
  });
  if (selected) {
    const path = Array.isArray(selected) ? selected[0] : selected;
    tmplAudioFile = path;
    templateAudioPath.value = path.split("/").pop() || path;
    try {
      tmplDuration = await invoke<number>("get_audio_duration", { path });
      tmplViewStart = 0;
      tmplViewEnd = tmplDuration;
      tmplSelStart = 0;
      tmplSelEnd = 0;
      await tmplLoadWaveform();
    } catch (e) {
      templateTimeInfo.textContent = `Error loading audio: ${e}`;
    }
  }
});

// Jump to time
btnTemplateJump.addEventListener("click", () => {
  const val = templateJumpTime.value.trim();
  const parts = val.split(":").map(Number);
  let secs = 0;
  if (parts.length === 2) secs = parts[0] * 60 + parts[1];
  else if (parts.length === 1) secs = parts[0];
  else return;
  if (isNaN(secs)) return;
  const viewDur = tmplViewEnd - tmplViewStart;
  tmplViewStart = Math.max(0, secs - viewDur / 2);
  tmplViewEnd = Math.min(tmplDuration, tmplViewStart + viewDur);
  tmplLoadWaveform();
});

// Zoom buttons
btnTemplateZoomIn.addEventListener("click", () => {
  const mid = (tmplViewStart + tmplViewEnd) / 2;
  const dur = (tmplViewEnd - tmplViewStart) * 0.5;
  tmplViewStart = Math.max(0, mid - dur / 2);
  tmplViewEnd = Math.min(tmplDuration, mid + dur / 2);
  tmplLoadWaveform();
  tmplDrawWaveform();
});

btnTemplateZoomOut.addEventListener("click", () => {
  const mid = (tmplViewStart + tmplViewEnd) / 2;
  const dur = Math.min(tmplDuration, (tmplViewEnd - tmplViewStart) * 2);
  tmplViewStart = Math.max(0, mid - dur / 2);
  tmplViewEnd = Math.min(tmplDuration, mid + dur / 2);
  tmplLoadWaveform();
  tmplDrawWaveform();
});

btnTemplateZoomFit.addEventListener("click", () => {
  tmplViewStart = 0;
  tmplViewEnd = tmplDuration;
  tmplLoadWaveform();
});

btnTemplateZoomSel.addEventListener("click", () => {
  if (tmplSelEnd <= tmplSelStart) return;
  const pad = (tmplSelEnd - tmplSelStart) * 0.3;
  tmplViewStart = Math.max(0, tmplSelStart - pad);
  tmplViewEnd = Math.min(tmplDuration, tmplSelEnd + pad);
  tmplLoadWaveform();
});

function tmplStopPlayback() {
  if (tmplPlayingSource) {
    try { tmplPlayingSource.stop(); } catch {}
    tmplPlayingSource = null;
  }
  btnTemplatePlay.textContent = "Play Selection";
  tmplDrawWaveform();
}

function tmplPlaySamples(samples: number[] | Float32Array, startSec: number) {
  if (!tmplAudioCtx) tmplAudioCtx = new AudioContext();
  const buffer = tmplAudioCtx.createBuffer(1, samples.length, 16000);
  const channelData = buffer.getChannelData(0);
  for (let i = 0; i < samples.length; i++) channelData[i] = samples[i];
  const source = tmplAudioCtx.createBufferSource();
  source.buffer = buffer;
  source.connect(tmplAudioCtx.destination);
  source.start();
  tmplPlayingSource = source;
  tmplPlayStartCtx = tmplAudioCtx.currentTime;
  tmplPlayStartSec = startSec;
  btnTemplatePlay.textContent = "Stop";
  source.onended = () => {
    if (tmplPlayingSource === source) tmplStopPlayback();
  };
  if (!tmplPlayheadRAF) tmplAnimatePlayhead();
}

async function tmplTogglePlaySelection() {
  if (tmplPlayingSource) { tmplStopPlayback(); return; }
  if (tmplSelEnd <= tmplSelStart || !tmplAudioFile) return;
  try {
    const samples = await invoke<number[]>("get_audio_region_pcm", {
      path: tmplAudioFile,
      startSecs: tmplSelStart,
      endSecs: tmplSelEnd,
    });
    tmplPlaySamples(samples, tmplSelStart);
  } catch (e) {
    templateTimeInfo.textContent = `Playback error: ${e}`;
  }
}

btnTemplatePlay.addEventListener("click", tmplTogglePlaySelection);

// Space bar to play/stop when template modal open
window.addEventListener("keydown", (e) => {
  if (templateModal.classList.contains("hidden")) return;
  const target = e.target as HTMLElement;
  if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
  if (e.code === "Space") {
    e.preventDefault();
    tmplTogglePlaySelection();
  }
});

// Save template
btnTemplateSave.addEventListener("click", async () => {
  const name = templateNameInput.value.trim();
  if (!name) { templateTimeInfo.textContent = "Enter a template name"; return; }
  if (tmplSelEnd <= tmplSelStart) { templateTimeInfo.textContent = "Select a region first"; return; }
  if (!tmplAudioFile) return;

  try {
    const result = await invoke<string>("save_audio_template", {
      name,
      path: tmplAudioFile,
      startSecs: tmplSelStart,
      endSecs: tmplSelEnd,
    });
    templateTimeInfo.textContent = result;
    templateNameInput.value = "";
    tmplRefreshList();
  } catch (e) {
    templateTimeInfo.textContent = `Save error: ${e}`;
  }
});

let tmplMatches: TemplateMatchResult[] = [];
let tmplMatchSelected: boolean[] = [];

function tmplRenderMatches(matches: TemplateMatchResult[]) {
  tmplMatches = matches;
  tmplMatchSelected = matches.map(() => true);
  const wrap = document.getElementById("template-matches-wrap")!;
  const list = document.getElementById("template-matches-list")!;
  const count = document.getElementById("template-matches-count")!;
  count.textContent = String(matches.length);
  if (matches.length === 0) { wrap.style.display = "none"; return; }
  wrap.style.display = "block";
  list.innerHTML = matches.map((m, i) => `
    <div class="template-item" data-idx="${i}">
      <div class="template-item-info">
        <input type="checkbox" class="tmpl-match-chk" data-idx="${i}" checked />
        <span class="template-item-name">${tmplFormatTime(m.time_secs)}</span>
        <span class="template-item-dur">${(m.confidence * 100).toFixed(0)}%</span>
        <input type="text" class="form-input tmpl-match-title" data-idx="${i}" placeholder="Chapter title..." style="padding:2px 6px;font-size:0.75rem;width:180px;" />
      </div>
      <div>
        <button class="small tmpl-match-play" data-idx="${i}">Play</button>
        <button class="small tmpl-match-jump" data-idx="${i}">Show</button>
      </div>
    </div>
  `).join("");

  list.querySelectorAll(".tmpl-match-chk").forEach(el => {
    el.addEventListener("change", (e) => {
      const idx = +(e.target as HTMLElement).dataset.idx!;
      tmplMatchSelected[idx] = (e.target as HTMLInputElement).checked;
    });
  });
  list.querySelectorAll(".tmpl-match-play").forEach(el => {
    el.addEventListener("click", async () => {
      const idx = +(el as HTMLElement).dataset.idx!;
      const t = tmplMatches[idx].time_secs;
      if (tmplPlayingSource) { tmplStopPlayback(); return; }
      const samples = await invoke<number[]>("get_audio_region_pcm", {
        path: tmplAudioFile,
        startSecs: Math.max(0, t - 0.5),
        endSecs: Math.min(tmplDuration, t + 2.5),
      });
      tmplPlaySamples(samples, Math.max(0, t - 0.5));
    });
  });
  list.querySelectorAll(".tmpl-match-jump").forEach(el => {
    el.addEventListener("click", () => {
      const idx = +(el as HTMLElement).dataset.idx!;
      const t = tmplMatches[idx].time_secs;
      const viewDur = Math.max(4, tmplViewEnd - tmplViewStart);
      tmplViewStart = Math.max(0, t - viewDur / 2);
      tmplViewEnd = Math.min(tmplDuration, tmplViewStart + viewDur);
      tmplLoadWaveform();
    });
  });
}

document.getElementById("btn-tmpl-matches-all")?.addEventListener("click", () => {
  tmplMatchSelected = tmplMatches.map(() => true);
  document.querySelectorAll<HTMLInputElement>(".tmpl-match-chk").forEach(c => c.checked = true);
});
document.getElementById("btn-tmpl-matches-none")?.addEventListener("click", () => {
  tmplMatchSelected = tmplMatches.map(() => false);
  document.querySelectorAll<HTMLInputElement>(".tmpl-match-chk").forEach(c => c.checked = false);
});
document.getElementById("btn-tmpl-matches-save")?.addEventListener("click", async () => {
  if (!tmplAudioFile) { templateTimeInfo.textContent = "No audio file"; return; }
  const titles: string[] = [];
  document.querySelectorAll<HTMLInputElement>(".tmpl-match-title").forEach(el => {
    titles[+el.dataset.idx!] = el.value.trim();
  });
  const selected = tmplMatches
    .map((m, i) => ({ ...m, title: titles[i] || `Chapter ${i + 1}`, keep: tmplMatchSelected[i] }))
    .filter(m => m.keep)
    .sort((a, b) => a.time_secs - b.time_secs);
  if (selected.length === 0) { templateTimeInfo.textContent = "No matches selected"; return; }
  // Force first chapter to 0:00
  const chapters = selected.map((m, i) => ({
    title: i === 0 ? (m.title || "Chapter 1") : m.title,
    start_secs: i === 0 ? 0 : m.time_secs,
    start_time: tmplFormatHHMMSS(i === 0 ? 0 : m.time_secs),
  }));
  // Write JSON next to audio
  const base = tmplAudioFile.replace(/\.[^.]+$/, "");
  const outPath = `${base}.chapters.json`;
  try {
    await invoke("write_text_file", { path: outPath, content: JSON.stringify(chapters, null, 2) });
    templateTimeInfo.textContent = `Saved ${chapters.length} chapters → ${outPath.split("/").pop()}`;
  } catch (e) {
    templateTimeInfo.textContent = `Save failed: ${e}`;
  }
});

function tmplFormatHHMMSS(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

async function tmplRefreshList() {
  const templates = await invoke<TemplateInfo[]>("list_audio_templates");
  if (templates.length === 0) {
    templateListItems.innerHTML = '<div style="font-size:0.8rem;color:var(--text-muted);padding:8px 0;">No templates saved yet</div>';
    return;
  }
  templateListItems.innerHTML = templates.map(t => `
    <div class="template-item">
      <div class="template-item-info">
        <span class="template-item-name">${escapeHtml(t.name)}</span>
        <span class="template-item-dur">${t.duration_secs.toFixed(2)}s</span>
        <span class="template-item-dur">${escapeHtml(t.source_file.split("/").pop() || "")}</span>
      </div>
      <div>
        <button class="small btn-tmpl-play" data-name="${escapeHtml(t.name)}">Play</button>
        <button class="small btn-tmpl-test" data-name="${escapeHtml(t.name)}">Test</button>
        <button class="small danger btn-tmpl-delete" data-name="${escapeHtml(t.name)}">&times;</button>
      </div>
    </div>
  `).join("");

  templateListItems.querySelectorAll(".btn-tmpl-delete").forEach(btn => {
    btn.addEventListener("click", async () => {
      const name = (btn as HTMLElement).dataset.name!;
      await invoke("delete_audio_template", { name });
      tmplRefreshList();
    });
  });

  templateListItems.querySelectorAll(".btn-tmpl-play").forEach(btn => {
    btn.addEventListener("click", async () => {
      const name = (btn as HTMLElement).dataset.name!;
      if (tmplPlayingSource) { tmplStopPlayback(); return; }
      try {
        const samples = await invoke<number[]>("get_template_pcm", { name });
        tmplPlaySamples(samples, 0);
      } catch (e) {
        templateTimeInfo.textContent = `Template play error: ${e}`;
      }
    });
  });

  templateListItems.querySelectorAll(".btn-tmpl-test").forEach(btn => {
    btn.addEventListener("click", async () => {
      if (!tmplAudioFile) { templateTimeInfo.textContent = "Load an audio file first"; return; }
      const name = (btn as HTMLElement).dataset.name!;
      templateTimeInfo.textContent = `Scanning for "${name}" matches...`;
      try {
        const matches = await invoke<TemplateMatchResult[]>("find_template_matches", {
          audioPath: tmplAudioFile,
          templateName: name,
          threshold: parseFloat(templateThreshold.value),
        });
        tmplRenderMatches(matches);
        if (matches.length === 0) {
          templateTimeInfo.textContent = `No matches found for "${name}". Try lowering threshold.`;
        } else {
          templateTimeInfo.textContent = `Found ${matches.length} matches`;
        }
      } catch (e) {
        templateTimeInfo.textContent = `Match error: ${e}`;
      }
    });
  });
}

btnTemplateEditor.addEventListener("click", () => {
  tmplRefreshList();
  requestAnimationFrame(() => {
    tmplSetupCanvas();
    tmplDrawWaveform();
    if (tmplAudioFile) tmplLoadWaveform();
  });
});

window.addEventListener("resize", () => {
  if (!templateModal.classList.contains("hidden")) {
    tmplSetupCanvas();
    tmplDrawWaveform();
  }
});

btnTemplateLoadChapters.addEventListener("click", async () => {
  const selected = await open({
    multiple: false,
    filters: [{ name: "Chapters", extensions: ["json", "srt", "vtt", "cue", "txt"] }],
  });
  if (!selected) return;
  const path = Array.isArray(selected) ? selected[0] : selected;
  try {
    const text = await invoke<string>("read_text_file", { path });
    tmplChapterMarkers = parseChapterMarkers(text, path);
    templateTimeInfo.textContent = `Loaded ${tmplChapterMarkers.length} chapter markers`;
    tmplDrawWaveform();
  } catch (e) {
    templateTimeInfo.textContent = `Failed to load chapters: ${e}`;
  }
});

function parseChapterMarkers(text: string, path: string): { time: number; title: string }[] {
  const ext = path.split(".").pop()?.toLowerCase() || "";
  const result: { time: number; title: string }[] = [];
  const toSecs = (h: number, m: number, s: number) => h * 3600 + m * 60 + s;
  if (ext === "json") {
    try {
      const data = JSON.parse(text);
      const arr = Array.isArray(data) ? data : (data.chapters || []);
      for (const c of arr) {
        const t = c.start_secs ?? c.start ?? (c.start_time ? parseTimeStr(c.start_time) : 0);
        result.push({ time: Number(t) || 0, title: c.title || c.name || "" });
      }
    } catch {}
  } else if (ext === "cue") {
    const lines = text.split("\n");
    let title = "";
    for (const line of lines) {
      const tm = line.match(/TITLE\s+"(.+)"/);
      if (tm) { title = tm[1]; continue; }
      const im = line.match(/INDEX\s+01\s+(\d+):(\d+):(\d+)/);
      if (im) {
        result.push({ time: toSecs(0, +im[1], +im[2] + +im[3] / 75), title });
        title = "";
      }
    }
  } else if (ext === "srt" || ext === "vtt") {
    const re = /(\d+):(\d+):(\d+)[.,](\d+)\s*-->/g;
    let m;
    while ((m = re.exec(text))) {
      result.push({ time: toSecs(+m[1], +m[2], +m[3] + +m[4] / 1000), title: "" });
    }
  } else {
    // txt: "MM:SS Title" or "HH:MM:SS Title" per line
    for (const line of text.split("\n")) {
      const m = line.match(/^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s+(.*)$/);
      if (m) result.push({ time: toSecs(+(m[1] || 0), +m[2], +m[3]), title: m[4] });
    }
  }
  return result;
}

function parseTimeStr(s: string): number {
  const parts = s.split(":").map(Number);
  if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
  if (parts.length === 2) return parts[0] * 60 + parts[1];
  return parts[0] || 0;
}

listen<{status: string; progress: number}>("template-match-progress", (event) => {
  templateTimeInfo.textContent = event.payload.status;
});

// === Align Chapters Modal ===
const alignModal = document.getElementById("align-modal")!;
const alignTitle = document.getElementById("align-title")!;
const alignStrips = document.getElementById("align-strips")!;
const btnAlignApply = document.getElementById("btn-align-apply")!;

interface AlignStrip {
  idx: number;
  chapter: Chapter;
  markerSecs: number;
  viewStart: number;
  viewEnd: number;
  canvas: HTMLCanvasElement;
  waveform: WaveformData | null;
  pipelineMarkers?: { rawLlm?: number; srtCorrected?: number; yamnet?: number };
}

let alignAudioPath = "";
let alignDuration = 0;
let alignStripsData: AlignStrip[] = [];
let alignPlayingSource: AudioBufferSourceNode | null = null;
let alignAudioCtx: AudioContext | null = null;
let alignDragging: { idx: number } | null = null;
let alignPlayheadRAF = 0;
let alignPlayStartCtx = 0;
let alignPlayStartSec = 0;
let alignPlayingIdx = -1;

alignModal.querySelector(".modal-close")!.addEventListener("click", () => {
  alignModal.classList.add("hidden");
  alignStopPlayback();
});
alignModal.addEventListener("click", (e) => {
  if (e.target === alignModal) { alignModal.classList.add("hidden"); alignStopPlayback(); }
});

function alignStopPlayback() {
  if (alignPlayingSource) { try { alignPlayingSource.stop(); } catch {} alignPlayingSource = null; }
  alignPlayingIdx = -1;
  if (alignPlayheadRAF) { cancelAnimationFrame(alignPlayheadRAF); alignPlayheadRAF = 0; }
  alignStripsData.forEach(s => alignDrawStrip(s));
}

function alignFormatTime(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = (secs % 60).toFixed(2);
  return `${m}:${s.padStart(5, "0")}`;
}

function alignFormatHMS(secs: number): string {
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = Math.floor(secs % 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

async function openAlignModal(audioPath: string, chapters: Chapter[], duration: number, snappedChapters?: ChapterWithSnap[]) {
  alignAudioPath = audioPath;
  alignDuration = duration || await invoke<number>("get_audio_duration", { path: audioPath });
  alignTitle.textContent = `Align Chapters — ${audioPath.split("/").pop()}`;
  alignModal.classList.remove("hidden");
  alignStripsData = [];
  alignStrips.innerHTML = "";

  // Legend for pipeline markers
  if (snappedChapters && snappedChapters.length > 0) {
    const legend = document.createElement("div");
    legend.className = "align-legend";
    legend.innerHTML = `<span style="color:#f44336;">━━</span> Raw LLM &nbsp; <span style="color:#4caf50;">━━</span> SRT Corrected &nbsp; <span style="color:#2196f3;">━━</span> YAMNet Snap &nbsp; <span style="color:#ff9800;">━━</span> Current`;
    alignStrips.appendChild(legend);
  }

  // Try to load previously aligned chapters
  const base = audioPath.replace(/\.[^.]+$/, "");
  const alignedPath = `${base}_aligned_chapters.json`;
  try {
    const exists = await invoke<boolean>("file_exists", { path: alignedPath });
    if (exists) {
      const text = await invoke<string>("read_text_file", { path: alignedPath });
      const aligned: Chapter[] = JSON.parse(text);
      if (aligned.length > 0) chapters = aligned;
    }
  } catch {}

  const WINDOW = 15; // ±15s around each boundary

  for (let i = 0; i < chapters.length; i++) {
    const ch = chapters[i];
    const nextCh = chapters[i + 1];
    const markerSecs = ch.start_secs;
    const viewStart = Math.max(0, markerSecs - WINDOW);
    const viewEnd = Math.min(alignDuration, markerSecs + WINDOW);

    const stripEl = document.createElement("div");
    stripEl.className = "align-strip";
    stripEl.innerHTML = `
      <div class="align-strip-header">
        <span class="align-strip-title">${i === 0 ? "(Start)" : ""} ${escapeHtml(ch.title)}${nextCh ? ` → ${escapeHtml(nextCh.title)}` : ""}</span>
        <span class="align-strip-time" id="align-time-${i}">${alignFormatHMS(markerSecs)}</span>
      </div>
      <div class="align-strip-info" id="align-info-${i}">Drag the marker to adjust boundary. Click to set.</div>
      <div class="align-canvas-wrap">
        <canvas id="align-canvas-${i}" width="850" height="80"></canvas>
      </div>
      <div class="align-btns">
        <button class="small align-play" data-idx="${i}">Play</button>
        <button class="small align-zoom-in" data-idx="${i}">Zoom In</button>
        <button class="small align-zoom-out" data-idx="${i}">Zoom Out</button>
      </div>
    `;
    alignStrips.appendChild(stripEl);

    const canvas = document.getElementById(`align-canvas-${i}`) as HTMLCanvasElement;
    const pipelineMarkers = snappedChapters && snappedChapters[i] ? {
      rawLlm: chkFirstZero.checked && i === 0 ? 0 : snappedChapters[i].raw_llm_secs,
      srtCorrected: snappedChapters[i].srt_corrected_secs ?? undefined,
      yamnet: snappedChapters[i].yamnet_secs ?? undefined,
    } : undefined;
    const strip: AlignStrip = { idx: i, chapter: ch, markerSecs, viewStart, viewEnd, canvas, waveform: null, pipelineMarkers };
    alignStripsData.push(strip);

    // HiDPI
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const cssW = rect.width || 850;
    const cssH = rect.height || 80;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.getContext("2d")!.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Mouse handlers
    canvas.addEventListener("mousedown", (e) => {
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const secs = strip.viewStart + (x / r.width) * (strip.viewEnd - strip.viewStart);
      strip.markerSecs = Math.max(strip.viewStart, Math.min(strip.viewEnd, secs));
      alignDragging = { idx: i };
      alignUpdateStripTime(strip);
      alignDrawStrip(strip);
    });

    canvas.addEventListener("mousemove", (e) => {
      if (!alignDragging || alignDragging.idx !== i) return;
      const r = canvas.getBoundingClientRect();
      const x = e.clientX - r.left;
      const secs = strip.viewStart + (x / r.width) * (strip.viewEnd - strip.viewStart);
      strip.markerSecs = Math.max(strip.viewStart, Math.min(strip.viewEnd, secs));
      alignUpdateStripTime(strip);
      alignDrawStrip(strip);
    });

    // Load waveform
    alignLoadStrip(strip);
  }

  // Global mouseup
  window.addEventListener("mouseup", () => { alignDragging = null; });

  // Button handlers
  alignStrips.querySelectorAll(".align-play").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = +(btn as HTMLElement).dataset.idx!;
      alignPlayAt(idx);
    });
  });

  alignStrips.querySelectorAll(".align-zoom-in").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = +(btn as HTMLElement).dataset.idx!;
      const s = alignStripsData[idx];
      const center = s.markerSecs;
      const dur = (s.viewEnd - s.viewStart) * 0.5;
      s.viewStart = Math.max(0, center - dur / 2);
      s.viewEnd = Math.min(alignDuration, center + dur / 2);
      alignLoadStrip(s);
    });
  });

  alignStrips.querySelectorAll(".align-zoom-out").forEach(btn => {
    btn.addEventListener("click", () => {
      const idx = +(btn as HTMLElement).dataset.idx!;
      const s = alignStripsData[idx];
      const center = s.markerSecs;
      const dur = Math.min(alignDuration, (s.viewEnd - s.viewStart) * 2);
      s.viewStart = Math.max(0, center - dur / 2);
      s.viewEnd = Math.min(alignDuration, center + dur / 2);
      alignLoadStrip(s);
    });
  });
}

function alignUpdateStripTime(strip: AlignStrip) {
  const el = document.getElementById(`align-time-${strip.idx}`);
  if (el) el.textContent = alignFormatHMS(strip.markerSecs);
  const info = document.getElementById(`align-info-${strip.idx}`);
  if (info) info.textContent = `${alignFormatTime(strip.markerSecs)} (${strip.markerSecs > strip.chapter.start_secs ? "+" : ""}${(strip.markerSecs - strip.chapter.start_secs).toFixed(2)}s from original)`;
}

async function alignLoadStrip(strip: AlignStrip) {
  const cssW = strip.canvas.getBoundingClientRect().width || 850;
  try {
    strip.waveform = await invoke<WaveformData>("get_waveform_peaks", {
      path: alignAudioPath,
      startSecs: strip.viewStart,
      endSecs: strip.viewEnd,
      numPoints: Math.round(cssW),
    });
    alignDrawStrip(strip);
  } catch (e) {
    console.warn("Failed to load waveform for strip", strip.idx, e);
  }
}

function alignDrawStrip(strip: AlignStrip) {
  const ctx = strip.canvas.getContext("2d")!;
  const rect = strip.canvas.getBoundingClientRect();
  const w = rect.width || 850;
  const h = rect.height || 80;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0a0a1a";
  ctx.fillRect(0, 0, w, h);

  if (!strip.waveform) return;

  // Waveform
  const mid = h / 2;
  const scale = mid * 0.9;
  const stepX = w / strip.waveform.peaks_max.length;
  ctx.strokeStyle = "#00b0f0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < strip.waveform.peaks_max.length; i++) {
    const x = i * stepX;
    const yMax = mid - strip.waveform.peaks_max[i] * scale;
    const yMin = mid - strip.waveform.peaks_min[i] * scale;
    ctx.moveTo(x, yMax);
    ctx.lineTo(x, yMin);
  }
  ctx.stroke();

  // Center line
  ctx.strokeStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();

  // Original position (ghost line)
  const origX = ((strip.chapter.start_secs - strip.viewStart) / (strip.viewEnd - strip.viewStart)) * w;
  if (origX >= 0 && origX <= w && Math.abs(strip.markerSecs - strip.chapter.start_secs) > 0.1) {
    ctx.strokeStyle = "rgba(255, 152, 0, 0.3)";
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(origX, 0); ctx.lineTo(origX, h); ctx.stroke();
    ctx.setLineDash([]);
  }

  // Pipeline stage markers (colored lines)
  if (strip.pipelineMarkers) {
    const stages: { secs: number | undefined; color: string; label: string }[] = [
      { secs: strip.pipelineMarkers.rawLlm, color: "rgba(244, 67, 54, 0.6)", label: "R" },
      { secs: strip.pipelineMarkers.srtCorrected, color: "rgba(76, 175, 80, 0.6)", label: "S" },
      { secs: strip.pipelineMarkers.yamnet, color: "rgba(33, 150, 243, 0.6)", label: "Y" },
    ];
    for (const stage of stages) {
      if (stage.secs == null) continue;
      const sx = ((stage.secs - strip.viewStart) / (strip.viewEnd - strip.viewStart)) * w;
      if (sx < 0 || sx > w) continue;
      ctx.strokeStyle = stage.color;
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, h); ctx.stroke();
      ctx.setLineDash([]);
      // Label at top
      ctx.fillStyle = stage.color;
      ctx.font = "bold 9px monospace";
      ctx.fillText(stage.label, sx + 2, 10);
    }
  }

  // Marker line
  const markerX = ((strip.markerSecs - strip.viewStart) / (strip.viewEnd - strip.viewStart)) * w;
  if (markerX >= 0 && markerX <= w) {
    ctx.strokeStyle = "#ff9800";
    ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(markerX, 0); ctx.lineTo(markerX, h); ctx.stroke();
    // Handle grips
    ctx.fillStyle = "#ff9800";
    ctx.fillRect(markerX - 4, 0, 8, 10);
    ctx.fillRect(markerX - 4, h - 10, 8, 10);
  }

  // Playhead
  if (alignPlayingSource && alignAudioCtx && alignPlayingIdx === strip.idx) {
    const pos = alignPlayStartSec + (alignAudioCtx.currentTime - alignPlayStartCtx);
    const px = ((pos - strip.viewStart) / (strip.viewEnd - strip.viewStart)) * w;
    if (px >= 0 && px <= w) {
      ctx.strokeStyle = "#e94560";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    }
  }

  // Time labels
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = "10px monospace";
  const viewDur = strip.viewEnd - strip.viewStart;
  const labelStep = viewDur < 5 ? 0.5 : viewDur < 15 ? 1 : viewDur < 60 ? 5 : 10;
  let t = Math.ceil(strip.viewStart / labelStep) * labelStep;
  while (t < strip.viewEnd) {
    const x = ((t - strip.viewStart) / (strip.viewEnd - strip.viewStart)) * w;
    ctx.fillText(alignFormatTime(t), x + 2, h - 2);
    t += labelStep;
  }
}

async function alignPlayAt(idx: number) {
  alignStopPlayback();
  const strip = alignStripsData[idx];
  const start = Math.max(0, strip.markerSecs - getAlignBefore());
  const end = Math.min(alignDuration, strip.markerSecs + getAlignAfter());
  try {
    const samples = await invoke<number[]>("get_audio_region_pcm", {
      path: alignAudioPath,
      startSecs: start,
      endSecs: end,
    });
    if (!alignAudioCtx) alignAudioCtx = new AudioContext();
    const buffer = alignAudioCtx.createBuffer(1, samples.length, 16000);
    const cd = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) cd[i] = samples[i];
    const source = alignAudioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(alignAudioCtx.destination);
    source.start();
    alignPlayingSource = source;
    alignPlayStartCtx = alignAudioCtx.currentTime;
    alignPlayStartSec = start;
    alignPlayingIdx = idx;
    source.onended = () => { alignStopPlayback(); };

    function animAlign() {
      if (!alignPlayingSource || alignPlayingIdx !== idx) return;
      alignDrawStrip(strip);
      alignPlayheadRAF = requestAnimationFrame(animAlign);
    }
    animAlign();
  } catch (e) { console.warn("Align playback error:", e); }
}

// Apply aligned chapters
btnAlignApply.addEventListener("click", async () => {
  const chapters: Chapter[] = alignStripsData.map(s => ({
    title: s.chapter.title,
    start_time: alignFormatHMS(s.markerSecs),
    start_secs: s.markerSecs,
  }));

  // Write CUE
  try {
    await invoke("write_cue_file", { audioPath: alignAudioPath, chapters });
  } catch (e) { console.warn("CUE write failed:", e); }

  // Embed in FLAC
  if (chkEmbedFlac.checked) {
    try {
      await invoke("embed_chapters_in_flac", { req: { audio_path: alignAudioPath, chapters } });
    } catch (e) { console.warn("Embed failed:", e); }
  }

  // Save JSON
  const base = alignAudioPath.replace(/\.[^.]+$/, "");
  const jsonPath = `${base}_aligned_chapters.json`;
  try {
    await invoke("write_text_file", { path: jsonPath, content: JSON.stringify(chapters, null, 2) });
  } catch (e) { console.warn("JSON write failed:", e); }

  alert(`Saved ${chapters.length} aligned chapters (CUE${chkEmbedFlac.checked ? " + FLAC embedded" : ""} + JSON)`);
  alignModal.classList.add("hidden");
  alignStopPlayback();
});

// Init
loadGpuInfo();
loadSettingsIntoForm();
loadPreferences();
applyEngineUI();
renderQueue();
