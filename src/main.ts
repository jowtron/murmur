import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

interface ModelInfo {
  name: string;
  display_name: string;
  downloaded: boolean;
  filename: string;
  size_bytes: number;
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
  };
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
const selectModel = document.getElementById("select-model")! as HTMLSelectElement;
const selectFormat = document.getElementById("select-format")! as HTMLSelectElement;
const selectOutput = document.getElementById("select-output")! as HTMLSelectElement;
const selectThreads = document.getElementById("select-threads")! as HTMLSelectElement;
const selectConcurrent = document.getElementById("select-concurrent")! as HTMLSelectElement;
const chkAutoChapters = document.getElementById("chk-auto-chapters")! as HTMLInputElement;
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
        <div class="file-name">${escapeHtml(item.name)}</div>
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
        <span class="status status-${item.status}">
          ${item.status === "pending" ? "Pending" : ""}
          ${item.status === "queued" ? "Queued" : ""}
          ${item.status === "transcribing" ? `${Math.round(item.progress * 100)}%` : ""}
          ${item.status === "detecting" ? `<span id="detect-status-${item.id}">Detecting...</span>` : ""}
          ${item.status === "complete" ? "Done" : ""}
          ${item.status === "error" ? "Error" : ""}
          ${item.status === "cancelled" ? "Cancelled" : ""}
        </span>
      </div>
      <div class="actions">
        ${item.status === "transcribing" || item.status === "queued" || item.status === "detecting" ? `<button class="small danger btn-cancel" data-id="${item.id}">Cancel</button>` : ""}
        ${item.status === "error" || item.status === "cancelled" ? `<button class="small btn-retry" data-id="${item.id}">Retry</button>` : ""}
        ${item.status === "complete" ? `<button class="small btn-reprocess" data-id="${item.id}">Reprocess</button>` : ""}
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

  // Clickable chapters badge
  queueList.querySelectorAll(".chapters-badge.clickable").forEach((badge) => {
    badge.addEventListener("click", () => {
      const id = (badge as HTMLElement).dataset.id!;
      const item = queue.find((q) => q.id === id);
      if (item?.chapters) showChapterDetail(item);
    });
  });

  // Reprocess button — re-run chapter detection with current settings
  queueList.querySelectorAll(".btn-reprocess").forEach((btn) => {
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

  for (const path of paths) {
    const name = path.split("/").pop() || path;
    const item: QueueItem = {
      id: generateId(),
      path,
      name,
      duration: null,
      status: "pending",
      progress: 0,
      autoDetectChapters: autoChapters,
    };
    queue.push(item);

    invoke<number>("get_audio_duration", { path })
      .then((dur) => { item.duration = dur; renderQueue(); })
      .catch(() => {});
  }
  renderQueue();
}

async function checkModelAndPromptDownload(modelName: string): Promise<boolean> {
  const ready = await invoke<boolean>("is_model_ready", { name: modelName });
  if (ready) return true;

  const doDownload = confirm(
    `The model "${modelName}" is not downloaded yet.\n\nWould you like to download it now?`
  );
  if (!doDownload) return false;

  // Open model manager, refresh list, then wait for user to see it
  modelModal.classList.remove("hidden");
  await refreshModelList();

  // Wait for user to confirm by waiting for the model to exist
  // The download starts when user clicks Download in the modal
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
    item.error = (item.error || "") + " (No API key for chapter detection)";
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

async function transcribeItem(item: QueueItem) {
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
    const alreadyExists = await invoke<boolean>("check_transcription_exists", {
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

  const modelReady = await checkModelAndPromptDownload(model);
  if (!modelReady) return;

  btnTranscribeAll.disabled = true;

  const concurrent = parseInt(selectConcurrent.value);
  await invoke("set_concurrency", { permits: concurrent });

  // Update autoDetectChapters based on current checkbox state
  pendingItems.forEach((item) => {
    item.autoDetectChapters = chkAutoChapters.checked;
  });

  const promises = pendingItems.map((item) => transcribeItem(item));
  await Promise.all(promises);

  btnTranscribeAll.disabled = false;
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
async function refreshModelList() {
  const models = await invoke<ModelInfo[]>("list_models");
  const dir = await invoke<string>("get_models_dir");
  modelsDir.textContent = dir;

  modelList.innerHTML = models
    .map(
      (m) => `
    <div class="model-item" data-model="${m.name}">
      <span class="model-name">${m.display_name}</span>
      ${
        m.downloaded
          ? `<span class="model-status downloaded">Downloaded</span>`
          : `<button class="small btn-download-model" data-name="${m.name}">Download</button>`
      }
    </div>
  `
    )
    .join("");

  modelList.querySelectorAll(".btn-download-model").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const name = (btn as HTMLElement).dataset.name!;
      const item = btn.closest(".model-item")!;
      (btn as HTMLElement).outerHTML = `<div class="download-progress"><div class="progress-bar"><div class="fill model-dl-fill" data-model="${name}" style="width: 0%"></div></div><span class="model-dl-text" data-model="${name}">Starting...</span></div>`;

      try {
        await invoke("download_model", { name });
        const statusEl = item.querySelector(".download-progress");
        if (statusEl) statusEl.outerHTML = `<span class="model-status downloaded">Downloaded</span>`;
      } catch {
        const statusEl = item.querySelector(".download-progress");
        if (statusEl) statusEl.outerHTML = `<button class="small btn-download-model" data-name="${name}">Retry</button>`;
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
    filters: [{ name: "Audio", extensions: ["flac", "mp3", "wav", "ogg", "m4a", "aac", "wma", "opus"] }],
  });
  if (selected) {
    const paths = Array.isArray(selected) ? selected : [selected];
    await addFiles(paths);
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

btnModels.addEventListener("click", () => refreshModelList());

// Settings
document.getElementById("btn-save-settings")!.addEventListener("click", () => {
  saveSettings();
  settingsModal.classList.add("hidden");
});

function loadSettingsIntoForm() {
  const s = loadSettings();
  (document.getElementById("input-api-key") as HTMLInputElement).value = s.apiKey;
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
    if (data.status === "transcribing") {
      if (item.status !== "transcribing") {
        item.status = "transcribing";
        item.startedAt = Date.now();
      }
    }
    renderQueue();
  }
});

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
}

function loadPreferences() {
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
}

selectModel.addEventListener("change", savePreferences);
selectFormat.addEventListener("change", savePreferences);
selectThreads.addEventListener("change", savePreferences);
selectConcurrent.addEventListener("change", savePreferences);
chkAutoChapters.addEventListener("change", savePreferences);
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
renderQueue();
