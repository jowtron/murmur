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
}

const queue: QueueItem[] = [];
let customOutputDir: string | null = null;

// Settings
function loadSettings() {
  return {
    apiKey: localStorage.getItem("openrouter_api_key") || "",
    llmModel: localStorage.getItem("llm_model") || "google/gemini-2.0-flash-001",
    apiUrl: localStorage.getItem("api_url") || "https://openrouter.ai/api/v1/chat/completions",
    chapterPrompt: localStorage.getItem("chapter_prompt") || (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement)?.value || "",
    chapterOutputFormat: localStorage.getItem("chapter_output_format") || "txt",
  };
}

function saveSettings() {
  localStorage.setItem("openrouter_api_key", (document.getElementById("input-api-key") as HTMLInputElement).value);
  localStorage.setItem("llm_model", (document.getElementById("input-llm-model") as HTMLInputElement).value);
  localStorage.setItem("api_url", (document.getElementById("input-api-url") as HTMLInputElement).value);
  localStorage.setItem("chapter_prompt", (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement).value);
  localStorage.setItem("chapter_output_format", (document.getElementById("select-chapter-format") as HTMLSelectElement).value);
}

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
        ${item.chapters ? `<div class="chapters-badge">${item.chapters.length} chapters detected</div>` : ""}
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
          ${item.status === "detecting" ? "Detecting..." : ""}
          ${item.status === "complete" ? "Done" : ""}
          ${item.status === "error" ? "Error" : ""}
          ${item.status === "cancelled" ? "Cancelled" : ""}
        </span>
      </div>
      <div class="actions">
        ${item.status === "transcribing" || item.status === "queued" ? `<button class="small danger btn-cancel" data-id="${item.id}">Cancel</button>` : ""}
        ${item.status === "error" || item.status === "cancelled" ? `<button class="small btn-retry" data-id="${item.id}">Retry</button>` : ""}
        ${item.status !== "transcribing" && item.status !== "detecting" ? `<button class="small danger btn-remove" data-id="${item.id}">&times;</button>` : ""}
      </div>
    </div>
  `
    )
    .join("");

  // Attach handlers
  queueList.querySelectorAll(".btn-remove").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = (btn as HTMLElement).dataset.id!;
      const idx = queue.findIndex((q) => q.id === id);
      if (idx !== -1) { queue.splice(idx, 1); renderQueue(); }
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
}

// Elapsed time updater
setInterval(() => {
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

async function runChapterDetection(item: QueueItem): Promise<void> {
  const settings = loadSettings();
  if (!settings.apiKey) {
    item.error = (item.error || "") + " (No API key for chapter detection)";
    return;
  }

  item.status = "detecting";
  item.progress = 0.5;
  renderQueue();

  // Read the SRT output for this file
  const outputDir = customOutputDir || item.path.substring(0, item.path.lastIndexOf("/"));
  const stem = item.name.replace(/\.[^.]+$/, "");
  const srtPath = `${outputDir}/${stem}.srt`;

  try {
    const transcript = await invoke<string>("read_text_file", { path: srtPath });

    const chapters = await invoke<Chapter[]>("detect_chapters", {
      req: {
        transcript,
        api_key: settings.apiKey,
        model: settings.llmModel,
        base_url: settings.apiUrl,
        prompt: settings.chapterPrompt,
      },
    });

    item.chapters = chapters;

    // Save chapter file
    if (chapters.length > 0) {
      const chapterFormat = settings.chapterOutputFormat || "txt";
      let content: string;
      let ext: string;

      if (chapterFormat === "json") {
        content = JSON.stringify(chapters, null, 2);
        ext = "chapters.json";
      } else {
        content = chapters.map((ch) => `${ch.start_time} - ${ch.title}`).join("\n");
        ext = "chapters.txt";
      }

      const chapterPath = `${outputDir}/${stem}.${ext}`;
      await invoke("write_text_file", { path: chapterPath, content });
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

  item.status = "transcribing";
  item.progress = 0;
  item.error = undefined;
  item.modelUsed = model;
  item.startedAt = Date.now();
  item.elapsed = undefined;
  renderQueue();

  try {
    // Make sure format includes srt if auto-chapters is on
    let outputFormat = format;
    if (item.autoDetectChapters && format !== "all" && format !== "srt") {
      outputFormat = "all"; // Need SRT for chapter detection
    }

    await invoke("transcribe_file", {
      job: {
        id: item.id,
        path: item.path,
        model,
        output_format: outputFormat,
        output_dir: customOutputDir,
        threads,
      },
    });

    item.elapsed = Date.now() - (item.startedAt || Date.now());

    // Auto-detect chapters if enabled
    if (item.autoDetectChapters) {
      await runChapterDetection(item);
    }

    item.status = "complete";
    item.progress = 1.0;
  } catch (err: any) {
    item.elapsed = Date.now() - (item.startedAt || Date.now());
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
async function loadFeed(url: string) {
  const feedStatus = document.getElementById("feed-status")!;
  const feedEpisodes = document.getElementById("feed-episodes")!;

  feedStatus.textContent = "Loading feed...";
  feedEpisodes.innerHTML = "";

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

    feedStatus.textContent = `Found ${items.length} episodes. Select episodes to add:`;

    let html = "";
    items.forEach((item, i) => {
      const title = item.querySelector("title")?.textContent || `Episode ${i + 1}`;
      const date = item.querySelector("pubDate")?.textContent || "";
      const enclosure = item.querySelector("enclosure");
      const audioUrl = enclosure?.getAttribute("url") || "";

      if (audioUrl) {
        html += `
          <div class="feed-episode">
            <input type="checkbox" data-url="${escapeHtml(audioUrl)}" data-title="${escapeHtml(title)}" />
            <span class="ep-title">${escapeHtml(title)}</span>
            <span class="ep-date">${date ? new Date(date).toLocaleDateString() : ""}</span>
          </div>`;
      }
    });

    html += `<button class="primary feed-add-selected" id="btn-add-episodes">Add Selected Episodes</button>`;
    feedEpisodes.innerHTML = html;

    document.getElementById("btn-add-episodes")?.addEventListener("click", async () => {
      const checked = feedEpisodes.querySelectorAll('input[type="checkbox"]:checked') as NodeListOf<HTMLInputElement>;
      if (checked.length === 0) { alert("No episodes selected."); return; }

      feedStatus.textContent = `Added ${checked.length} episodes. Note: podcast episode download coming soon.`;
      feedModal.classList.add("hidden");
    });
  } catch (err) {
    feedStatus.textContent = `Failed to load feed: ${err}`;
  }
}

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

btnClearQueue.addEventListener("click", () => {
  const active = queue.some((q) => q.status === "transcribing" || q.status === "detecting");
  if (active && !confirm("There are active transcriptions. Clear anyway?")) return;
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
  (document.getElementById("input-llm-model") as HTMLInputElement).value = s.llmModel;
  (document.getElementById("input-api-url") as HTMLInputElement).value = s.apiUrl;
  if (s.chapterPrompt) (document.getElementById("input-chapter-prompt") as HTMLTextAreaElement).value = s.chapterPrompt;
  (document.getElementById("select-chapter-format") as HTMLSelectElement).value = s.chapterOutputFormat;
}

// Feed
document.getElementById("btn-load-feed")!.addEventListener("click", () => {
  const url = (document.getElementById("input-feed-url") as HTMLInputElement).value.trim();
  if (url) loadFeed(url);
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

      const chapters = await invoke<Chapter[]>("detect_chapters", {
        req: {
          transcript,
          api_key: settings.apiKey,
          model: settings.llmModel,
          base_url: settings.apiUrl,
          prompt: settings.chapterPrompt,
        },
      });

      if (chapters.length === 0) {
        chaptersResults.innerHTML += `<div class="chapter-file-header">${escapeHtml(fileName)} - No chapters detected</div>`;
        continue;
      }

      // Auto-save
      const dir = filePath.substring(0, filePath.lastIndexOf("/"));
      const stem = fileName.replace(/\.[^.]+$/, "");
      const chapterFormat = settings.chapterOutputFormat || "txt";

      let content: string;
      let ext: string;
      if (chapterFormat === "json") {
        content = JSON.stringify(chapters, null, 2);
        ext = "chapters.json";
      } else {
        content = chapters.map((ch) => `${ch.start_time} - ${ch.title}`).join("\n");
        ext = "chapters.txt";
      }

      const outPath = `${customOutputDir || dir}/${stem}.${ext}`;
      await invoke("write_text_file", { path: outPath, content });

      // Display
      let html = `<div class="chapter-file-header">${escapeHtml(fileName)} (${chapters.length} chapters) <span class="saved-badge">Saved: ${stem}.${ext}</span></div>`;
      html += chapters.map((ch) => `
        <div class="chapter-item">
          <span class="ch-time">${escapeHtml(ch.start_time)}</span>
          <span class="ch-title">${escapeHtml(ch.title)}</span>
        </div>`).join("");
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
    if (data.status === "transcribing") item.status = "transcribing";
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

// Init
loadGpuInfo();
loadSettingsIntoForm();
renderQueue();
