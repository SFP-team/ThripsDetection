const NAME_KEY = "thrips.annotator.name";
const SESSION_KEY = "thrips.annotator.session";

const state = {
  screen: "load",
  settings: null,
  batch: null,
  tile: null,
  plant: [],
  jobId: null,
  poll: null,
  draft: { tissue: null, injury: null },
  fromReview: false,
  plantFilter: "",
  skipPush: false,
  predictionRunId: null,
  predictionJobId: null,
  predictionPoll: null,
  predictionRun: null,
  plants: [],
  selectedPlant: null,
  plantDetail: null,
  selectedTile: null,
  plantView: null,
  highFrac: 0.2,
  midFrac: 0.3,
};

const TISSUE_NAME = {
  flush: "new growth",
  mature: "old leaves",
  tube: "no leaf",
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  if (response.headers.get("content-type")?.includes("text/csv")) {
    return response;
  }
  return response.json();
}

function pathFor() {
  if (state.screen === "guide") return "/how-to-annotate";
  if (state.screen === "predict") {
    const base = state.predictionRunId ? `/predict/${encodeURIComponent(state.predictionRunId)}` : "/predict";
    if (state.selectedPlant) return `${base}?plant=${encodeURIComponent(state.selectedPlant)}`;
    return base;
  }
  if (!state.batch || state.screen === "load" || state.screen === "progress") return "/";
  if (state.screen === "review") {
    const query = state.plantFilter ? `?plant=${encodeURIComponent(state.plantFilter)}` : "";
    return `/batches/${state.batch.id}/review${query}`;
  }
  const query = state.tile ? `?tile=${state.tile.id}` : "";
  return `/batches/${state.batch.id}/label${query}`;
}

function persist(replace = false) {
  const name = ($("annotator")?.value || "").trim();
  if (name) localStorage.setItem(NAME_KEY, name);
  if (state.batch && state.screen !== "guide" && state.screen !== "progress") {
    localStorage.setItem(SESSION_KEY, JSON.stringify({
      batchId: state.batch.id,
      tileId: state.tile?.id || null,
      screen: state.screen,
      plantFilter: state.plantFilter || "",
      fromReview: state.fromReview,
    }));
  }
  const url = pathFor();
  const next = `${location.origin}${url}`;
  if (next === location.href) return;
  const method = replace || state.skipPush ? "replaceState" : "pushState";
  history[method]({ screen: state.screen, batchId: state.batch?.id || null }, "", url);
}

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
  } catch {
    return null;
  }
}

function parseLocation() {
  const path = location.pathname.replace(/\/$/, "") || "/";
  if (path === "/how-to-annotate") return { screen: "guide" };
  if (path === "/predict") return { screen: "predict" };
  const predictMatch = path.match(/^\/predict\/([^/]+)$/);
  if (predictMatch) {
    const params = new URLSearchParams(location.search);
    return {
      screen: "predict",
      runId: decodeURIComponent(predictMatch[1]),
      plant: params.get("plant") || "",
    };
  }
  const match = path.match(/^\/batches\/(\d+)\/(label|review)$/);
  if (!match) return null;
  const params = new URLSearchParams(location.search);
  return {
    batchId: Number(match[1]),
    screen: match[2],
    tileId: params.get("tile") ? Number(params.get("tile")) : null,
    plantFilter: params.get("plant") || "",
  };
}

function show(screen, replace = false) {
  state.screen = screen;
  document.body.dataset.screen = screen;
  document.title = screen === "predict" ? "Predict visible plant damage" : "Tiles annotator for thrips detection";
  for (const id of ["load", "progress", "label", "review", "guide", "predict"]) {
    const node = $(`screen-${id}`);
    if (!node) continue;
    const visible = id === screen;
    node.hidden = !visible;
    node.classList.toggle("is-visible", visible);
  }
  const hasBatch = Boolean(state.batch);
  $("top-nav").hidden = screen === "progress";
  for (const item of document.querySelectorAll("#top-nav [data-needs-batch]")) {
    item.hidden = !hasBatch;
  }
  for (const button of document.querySelectorAll("#top-nav [data-go]")) {
    button.classList.toggle("active", button.dataset.go === screen);
  }
  if (hasBatch) {
    $("export-link").href = `/api/batches/${state.batch.id}/export`;
  }
  paintGpuControls();
  persist(replace);
}

function annotator() {
  return ($("annotator").value.trim() || localStorage.getItem(NAME_KEY) || "").trim();
}

function requireName() {
  const name = annotator();
  if (!name) {
    flash({ message: "Type your name first." });
    return "";
  }
  localStorage.setItem(NAME_KEY, name);
  $("annotator").value = name;
  return name;
}

function flash(error) {
  const toast = $("toast");
  toast.hidden = false;
  toast.textContent = error?.message || String(error);
  requestAnimationFrame(() => toast.classList.add("is-on"));
  clearTimeout(flash._t);
  flash._t = setTimeout(() => {
    toast.classList.remove("is-on");
  }, 3200);
}

function fadeImage(node, src) {
  if (!node || node.dataset.src === src) return;
  node.classList.add("is-switching");
  window.setTimeout(() => {
    node.onload = () => node.classList.remove("is-switching");
    node.src = src;
    node.dataset.src = src;
  }, 140);
}

function draftKey(tileId) {
  return `thrips.draft.${tileId}`;
}

function stashDraft() {
  if (!state.tile) return;
  sessionStorage.setItem(draftKey(state.tile.id), JSON.stringify(state.draft));
}

function resetDraft(fromLabel) {
  state.draft = {
    tissue: fromLabel?.tissue || null,
    injury: fromLabel?.injury || fromLabel?.label || null,
  };
}

function restoreDraft(tile) {
  const raw = sessionStorage.getItem(draftKey(tile.id));
  if (!raw) {
    resetDraft(tile.current_label);
    return;
  }
  try {
    state.draft = JSON.parse(raw);
  } catch {
    resetDraft(tile.current_label);
  }
}

function paintDraft() {
  const flush = state.draft.tissue === "flush";
  $("step-injury").classList.toggle("is-dim", !flush);
  if ($("draft-status")) $("draft-status").textContent = draftStatus();
  for (const button of document.querySelectorAll("[data-tissue]")) {
    button.classList.toggle("active", button.dataset.tissue === state.draft.tissue);
  }
  for (const button of document.querySelectorAll("[data-injury]")) {
    button.classList.toggle("active", flush && button.dataset.injury === state.draft.injury);
  }
}

function setProgress(step, title, detail) {
  $("progress-step").textContent = step;
  $("progress-title").textContent = title;
  $("progress-detail").textContent = detail || "";
  for (const item of document.querySelectorAll(".steps li")) {
    item.classList.toggle("active", item.dataset.step === step);
    const order = ["sending", "finding", "cutting", "leaves", "copying", "ready"];
    item.classList.toggle("done", order.indexOf(item.dataset.step) < order.indexOf(step));
  }
}

const STEP_TITLES = {
  sending: "Sending photos to the GPU",
  finding: "Finding the plant",
  cutting: "Cutting it out of the background",
  leaves: "Keeping only leaf squares",
  copying: "Opening tiles",
  ready: "Ready to label",
  error: "Something went wrong",
};

async function watchJob(jobId) {
  state.jobId = jobId;
  show("progress", true);
  if (state.poll) clearInterval(state.poll);
  const tick = async () => {
    const job = await api(`/api/jobs/${jobId}`);
    setProgress(job.step || "copying", STEP_TITLES[job.step] || job.step, job.detail);
    if (job.status === "done") {
      clearInterval(state.poll);
      await openBatch(job.batch_id);
    }
    if (job.status === "error") {
      clearInterval(state.poll);
      setProgress("error", "Something went wrong", job.detail);
      flash(job.detail);
    }
  };
  await tick();
  state.poll = setInterval(tick, 1500);
}

async function openBatch(batchId, tileId = null, options = {}) {
  stashDraft();
  if (options.fromReview != null) state.fromReview = options.fromReview;
  state.batch = await api(`/api/batches/${batchId}`);
  const query = tileId ? `tile_id=${tileId}` : "";
  const payload = await api(`/api/batches/${batchId}/next?${query}`);
  state.batch = payload.batch;
  if (payload.done && !tileId) {
    state.tile = null;
    state.plant = [];
    show("label", options.replace);
    renderDone();
    return;
  }
  state.tile = payload.tile;
  state.plant = payload.plant;
  restoreDraft(payload.tile);
  show("label", options.replace);
  renderLabel();
}

function renderDone() {
  $("label-work").hidden = true;
  $("label-done").hidden = false;
  $("label-meta").innerHTML = "<span>Batch complete</span><span>0 left</span>";
  $("progress-fill").style.width = "100%";
}

function tileTone(item) {
  if (item.tissue === "flush" && item.injury === "healthy") return "flush_healthy";
  if (item.tissue === "flush" && item.injury === "mild") return "flush_mild";
  if (item.tissue === "flush" && item.injury === "severe") return "flush_severe";
  if (item.tissue === "flush" && item.injury === "uncertain") return "flush_uncertain";
  if (item.tissue === "flush" && item.injury === "injured") return "flush_injured";
  if (item.tissue === "mature") return "mature";
  if (item.tissue === "tube" || item.injury === "skip" || item.label === "skip") return "skip";
  return item.injury || "";
}

function renderLabel() {
  if (!state.tile) {
    renderDone();
    return;
  }
  $("label-work").hidden = false;
  $("label-done").hidden = true;
  const counts = state.batch.counts;
  const plants = state.batch.images || [];
  const plantNo = plants.findIndex((item) => item.id === state.tile.image_id) + 1;
  const plantTiles = state.plant;
  const tileNo = plantTiles.findIndex((item) => item.id === state.tile.id) + 1;
  const pct = counts.tiles ? Math.round((counts.labeled / counts.tiles) * 100) : 0;
  $("label-meta").innerHTML = `
    <span>${state.tile.image} · plant ${plantNo} / ${plants.length} · tile ${tileNo} / ${plantTiles.length}</span>
    <span>${counts.labeled} labeled · ${counts.unlabeled} left · ${counts.flush_mild || 0} mild · ${counts.flush_severe || 0} severe</span>
  `;
  $("progress-fill").style.width = `${pct}%`;
  fadeImage($("tile-image"), `/media/tile/${state.tile.id}`);
  fadeImage($("context-image"), `/media/context/${state.tile.id}`);
  $("context-image").onerror = () => {
    fadeImage($("context-image"), `/media/crop/${state.tile.image_id}`);
  };
  paintDraft();
  $("filmstrip").innerHTML = plantTiles
    .map(
      (item) => `
      <button type="button" class="${item.id === state.tile.id ? "current" : ""} ${tileTone(item)}" data-tile="${item.id}" title="${TISSUE_NAME[item.tissue] || "unlabeled"} ${item.injury || ""}">
        <img src="/media/tile/${item.id}" alt="${item.tile}">
      </button>`
    )
    .join("");
  const currentThumb = $("filmstrip").querySelector(".current");
  if (currentThumb) currentThumb.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
}

function draftStatus() {
  if (!state.draft.tissue) return "Mark the tissue first.";
  if (state.draft.tissue !== "flush") return `${TISSUE_NAME[state.draft.tissue]}: no damage label needed.`;
  if (!state.draft.injury) return "New growth. Now mark healthy, mild, severe, or unsure.";
  if (!["healthy", "mild", "severe", "uncertain"].includes(state.draft.injury)) {
    return "Legacy label. Choose healthy, mild, severe, or unsure to update it.";
  }
  return `New growth: ${state.draft.injury}.`;
}

function canSaveDraft() {
  if (!state.draft.tissue) return false;
  if (state.draft.tissue !== "flush") return true;
  return ["healthy", "mild", "severe", "uncertain"].includes(state.draft.injury);
}

async function saveDraft() {
  if (!state.tile || !canSaveDraft()) return;
  const name = requireName();
  if (!name) return;
  const payload = await api("/api/label", {
    method: "POST",
    body: JSON.stringify({
      tile_id: state.tile.id,
      tissue: state.draft.tissue,
      injury: state.draft.injury,
      curl: null,
      annotator: name,
    }),
  });
  sessionStorage.removeItem(draftKey(state.tile.id));
  state.batch = payload.batch;
  if (state.fromReview) {
    state.fromReview = false;
    show("review");
    await renderReview();
    return;
  }
  if (payload.done) {
    state.tile = null;
    state.plant = [];
    renderDone();
    persist(true);
    return;
  }
  state.tile = payload.next;
  state.plant = payload.plant;
  restoreDraft(payload.next);
  renderLabel();
  persist(true);
}

async function chooseTissue(tissue) {
  if (!state.tile || state.screen !== "label") return;
  state.draft.tissue = tissue;
  if (tissue !== "flush") {
    state.draft.injury = "skip";
    paintDraft();
    await saveDraft();
    return;
  }
  state.draft.injury = null;
  paintDraft();
  stashDraft();
}

async function chooseInjury(injury) {
  if (!state.tile || state.draft.tissue !== "flush") return;
  state.draft.injury = injury;
  paintDraft();
  await saveDraft();
}

async function undo() {
  if (!state.batch) return;
  const payload = await api("/api/undo", {
    method: "POST",
    body: JSON.stringify({ batch_id: state.batch.id }),
  });
  state.batch = payload.batch;
  state.tile = payload.tile;
  state.plant = payload.plant;
  state.fromReview = false;
  resetDraft(payload.tile?.current_label);
  if (payload.tile) sessionStorage.removeItem(draftKey(payload.tile.id));
  show("label", true);
  renderLabel();
}

function reviewBucket(item) {
  if (item.tissue === "flush" && item.injury === "healthy") return "flush_healthy";
  if (item.tissue === "flush" && item.injury === "mild") return "flush_mild";
  if (item.tissue === "flush" && item.injury === "severe") return "flush_severe";
  if (item.tissue === "flush" && item.injury === "uncertain") return "flush_uncertain";
  if (item.tissue === "flush" && item.injury === "injured") return "flush_injured";
  if (item.tissue === "mature") return "mature";
  return "skip";
}

async function renderReview() {
  if (!state.batch) return;
  const imageId = $("review-plant").value || state.plantFilter;
  state.plantFilter = imageId || "";
  const query = imageId ? `image_id=${imageId}` : "";
  const payload = await api(`/api/batches/${state.batch.id}/review?${query}`);
  state.batch = payload.batch;
  const counts = state.batch.counts;
  $("review-counts").textContent =
    `${counts.flush_healthy || 0} healthy · ${counts.flush_mild || 0} mild · ${counts.flush_severe || 0} severe · ${counts.flush_uncertain || 0} unsure · ${counts.mature || 0} old leaves · ${counts.unlabeled} left`;
  const select = $("review-plant");
  select.innerHTML =
    `<option value="">All plants</option>` +
    (state.batch.images || [])
      .map((item) => `<option value="${item.id}">${item.filename}</option>`)
      .join("");
  select.value = state.plantFilter;
  const legacy = $("legacy-review");
  if (legacy) legacy.hidden = !(counts.flush_injured);
  for (const name of ["flush_healthy", "flush_mild", "flush_severe", "flush_uncertain", "flush_injured", "mature", "skip"]) {
    const column = document.querySelector(`[data-col="${name}"]`);
    if (!column) continue;
    const tiles = payload.tiles.filter((item) => reviewBucket(item) === name);
    column.innerHTML = tiles
      .map((item) => {
        return `<button type="button" data-tile="${item.id}" title="${item.image} ${item.tissue || ""} ${item.injury || ""}"><img src="/media/tile/${item.id}" alt="${item.tile}"></button>`;
      })
      .join("");
  }
}

function gpuReady() {
  return Boolean(state.settings?.gpu_ready);
}

function paintGpuControls() {
  const ready = gpuReady();
  const card = $("form-gpu");
  if (card) {
    card.classList.toggle("is-off", !ready);
    $("gpu-path-btn").disabled = !ready;
    $("gpu-path").disabled = !ready;
    $("gpu-folder-input").disabled = !ready;
    $("gpu-hint").textContent = ready
      ? "This computer sends JPGs to the GPU. Tiles come back. Then you label here."
      : "GPU password is not set on this computer. Ask for a local settings file.";
  }
  const exportGpu = $("export-gpu");
  if (exportGpu) {
    exportGpu.hidden = !ready || !state.batch || ["progress", "load", "guide", "predict"].includes(state.screen);
    exportGpu.disabled = !ready || !state.batch;
  }
  paintPredictGpu();
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  const name = localStorage.getItem(NAME_KEY) || "";
  $("annotator").value = name;
  if ($("predict-annotator")) $("predict-annotator").value = name;
  paintGpuControls();
}

async function loadSessions() {
  const sessions = await api("/api/sessions");
  $("session-list").hidden = sessions.length === 0;
  $("start-btn").disabled = sessions.length === 0;
  if (!sessions.length) {
    $("home-status").textContent = "No session yet. Start from local tiles or GPU photos.";
    $("batches").innerHTML = "";
    return sessions;
  }
  const latest = sessions[0];
  $("home-status").textContent =
    `${latest.name} · ${latest.counts.labeled} labeled · ${latest.counts.unlabeled} left. Saved in ${latest.folder || "this folder"}.`;
  $("batches").innerHTML = sessions
    .map((item) => {
      const left = item.counts.unlabeled;
      return `<li><button type="button" data-batch="${item.batch_id}">
        <span>${item.name}</span>
        <span class="mono">${item.counts.tiles} tiles · ${left} left</span>
      </button></li>`;
    })
    .join("");
  return sessions;
}

async function startWork(tileId = null, options = {}) {
  const name = requireName();
  if (!name) return;
  const sessions = await loadSessions();
  if (!sessions.length) {
    flash({ message: "Upload a folder of tiles first." });
    return;
  }
  const remembered = readSession();
  const batchId = options.batchId || sessions[0].batch_id;
  const known = sessions.some((item) => item.batch_id === batchId);
  const resumeTile =
    remembered?.batchId === batchId ? tileId || remembered.tileId : tileId;
  await openBatch(known ? batchId : sessions[0].batch_id, resumeTile, options);
}

async function createSessionFromPath() {
  const name = requireName();
  if (!name) return;
  const path = $("session-path").value.trim();
  if (!path) {
    flash({ message: "Choose a folder or paste a path." });
    return;
  }
  show("progress", true);
  setProgress("copying", "Copying folder into this app", path);
  const job = await api("/api/sessions/from-path", {
    method: "POST",
    body: JSON.stringify({
      annotator: name,
      name: $("session-name").value.trim(),
      path,
    }),
  });
  await openBatch(job.batch_id);
}

async function createSessionFromFiles(fileList) {
  const name = requireName();
  if (!name) return;
  const files = [...fileList];
  if (!files.length) {
    flash({ message: "Choose a folder of tiles." });
    return;
  }
  const data = new FormData();
  data.append("annotator", name);
  data.append("name", $("session-name").value.trim() || files[0].webkitRelativePath.split("/")[0] || "");
  for (const file of files) {
    data.append("files", file);
    data.append("relpaths", file.webkitRelativePath || file.name);
  }
  show("progress", true);
  setProgress("copying", "Copying folder into this app", `${files.length} files`);
  const response = await fetch("/api/sessions/upload", { method: "POST", body: data });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  const job = await response.json();
  await openBatch(job.batch_id);
}

async function createGpuSessionFromPath() {
  const name = requireName();
  if (!name) return;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return;
  }
  const path = $("gpu-path").value.trim();
  if (!path) {
    flash({ message: "Choose raw photos or paste a folder path." });
    return;
  }
  const job = await api("/api/sessions/gpu", {
    method: "POST",
    body: JSON.stringify({
      annotator: name,
      name: $("session-name").value.trim(),
      path,
    }),
  });
  await watchJob(job.job_id);
}

async function createGpuSessionFromFiles(fileList) {
  const name = requireName();
  if (!name) return;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return;
  }
  const files = [...fileList];
  if (!files.length) {
    flash({ message: "Choose a folder of raw photos." });
    return;
  }
  const data = new FormData();
  data.append("annotator", name);
  data.append("name", $("session-name").value.trim() || files[0].webkitRelativePath.split("/")[0] || "");
  for (const file of files) {
    data.append("files", file);
    data.append("relpaths", file.webkitRelativePath || file.name);
  }
  show("progress", true);
  setProgress("sending", "Sending photos to the GPU", `${files.length} files`);
  const response = await fetch("/api/sessions/gpu-upload", { method: "POST", body: data });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  const job = await response.json();
  await watchJob(job.job_id);
}

async function exportToGpu() {
  if (!state.batch) return;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return;
  }
  const result = await api(`/api/batches/${state.batch.id}/export-gpu`, { method: "POST" });
  flash({ message: result.detail || "Export sent to the GPU." });
}

async function readAllDirectoryEntries(reader) {
  const entries = [];
  while (true) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    entries.push(...batch);
  }
  return entries;
}

async function filesFromDrop(event) {
  const items = [...(event.dataTransfer?.items || [])];
  if (!items.length) return [...(event.dataTransfer?.files || [])];
  const files = [];
  const walk = async (entry, prefix) => {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      Object.defineProperty(file, "webkitRelativePath", { value: `${prefix}${entry.name}` });
      files.push(file);
      return;
    }
    if (!entry.isDirectory) return;
    const children = await readAllDirectoryEntries(entry.createReader());
    for (const child of children) {
      await walk(child, `${prefix}${entry.name}/`);
    }
  };
  for (const item of items) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) await walk(entry, "");
  }
  return files.length ? files : [...(event.dataTransfer?.files || [])];
}

$("form-start").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const session = readSession();
    await startWork(session?.tileId || null);
  } catch (error) {
    flash(error);
  }
});

$("form-session").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await createSessionFromPath();
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("folder-input").addEventListener("change", async (event) => {
  try {
    await createSessionFromFiles(event.target.files);
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("drop-zone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("drop-zone").classList.add("is-over");
});
$("drop-zone").addEventListener("dragleave", () => {
  $("drop-zone").classList.remove("is-over");
});
$("drop-zone").addEventListener("drop", async (event) => {
  event.preventDefault();
  $("drop-zone").classList.remove("is-over");
  try {
    const files = await filesFromDrop(event);
    await createSessionFromFiles(files);
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("form-gpu").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await createGpuSessionFromPath();
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("gpu-folder-input").addEventListener("change", async (event) => {
  try {
    await createGpuSessionFromFiles(event.target.files);
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("gpu-drop-zone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("gpu-drop-zone").classList.add("is-over");
});
$("gpu-drop-zone").addEventListener("dragleave", () => {
  $("gpu-drop-zone").classList.remove("is-over");
});
$("gpu-drop-zone").addEventListener("drop", async (event) => {
  event.preventDefault();
  $("gpu-drop-zone").classList.remove("is-over");
  try {
    const files = await filesFromDrop(event);
    await createGpuSessionFromFiles(files);
  } catch (error) {
    flash(error);
    show("load");
  }
});

$("export-gpu").addEventListener("click", async () => {
  try {
    await exportToGpu();
  } catch (error) {
    flash(error);
  }
});

$("batches").addEventListener("click", (event) => {
  const button = event.target.closest("[data-batch]");
  if (!button) return;
  startWork(null, { batchId: Number(button.dataset.batch) }).catch(flash);
});

$("filmstrip").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tile]");
  if (button) openBatch(state.batch.id, Number(button.dataset.tile), { fromReview: false, replace: true }).catch(flash);
});

$("top-nav").addEventListener("click", (event) => {
  const go = event.target.closest("[data-go]")?.dataset.go;
  if (go === "load") {
    state.fromReview = false;
    show("load");
    loadSessions().catch(flash);
  }
  if (go === "guide") {
    show("guide");
    window.scrollTo(0, 0);
  }
  if (go === "predict") {
    state.fromReview = false;
    show("predict");
    if (state.predictionRunId && state.predictionRun?.status === "ready") {
      if (state.selectedPlant && state.plantDetail) {
        showPredictPanel("detail");
        requestAnimationFrame(drawPredictOverlay);
      } else {
        renderPredictTable();
      }
    } else if (state.predictionJobId) {
      showPredictPanel("progress");
    } else {
      initPredictPage().catch(flash);
    }
  }
  if (go === "label" && state.batch) {
    openBatch(state.batch.id, state.tile?.id || readSession()?.tileId || null, {
      fromReview: false,
    }).catch(flash);
  }
  if (go === "review" && state.batch) {
    state.fromReview = false;
    show("review");
    renderReview().catch(flash);
  }
});

$("review-plant").addEventListener("change", () => {
  state.plantFilter = $("review-plant").value;
  renderReview().then(() => persist(true)).catch(flash);
});

$("screen-review").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tile]");
  if (!button) return;
  openBatch(state.batch.id, Number(button.dataset.tile), { fromReview: true }).catch(flash);
});

document.querySelector("#step-tissue").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tissue]");
  if (button) chooseTissue(button.dataset.tissue).catch(flash);
});

document.querySelector("#step-injury").addEventListener("click", (event) => {
  const injury = event.target.closest("[data-injury]");
  if (injury) chooseInjury(injury.dataset.injury).catch(flash);
});

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
  if (typing || state.screen !== "label") return;
  const key = event.key;
  if (key === "Escape") {
    resetDraft(state.tile?.current_label);
    if (state.tile) sessionStorage.removeItem(draftKey(state.tile.id));
    paintDraft();
    if ($("draft-status")) $("draft-status").textContent = draftStatus();
    return;
  }
  if (key === "z" || key === "Z") {
    undo().catch(flash);
    return;
  }
  if (key === "ArrowRight" && state.plant.length && state.tile) {
    const index = state.plant.findIndex((item) => item.id === state.tile.id);
    const next = state.plant[index + 1];
    if (next) openBatch(state.batch.id, next.id, { fromReview: false, replace: true }).catch(flash);
    return;
  }
  if (key === "ArrowLeft" && state.plant.length && state.tile) {
    const index = state.plant.findIndex((item) => item.id === state.tile.id);
    const prev = state.plant[index - 1];
    if (prev) openBatch(state.batch.id, prev.id, { fromReview: false, replace: true }).catch(flash);
    else undo().catch(flash);
    return;
  }
  if (!state.draft.tissue) {
    if (key === "1") chooseTissue("flush").catch(flash);
    if (key === "2") chooseTissue("mature").catch(flash);
    if (key === "3") chooseTissue("tube").catch(flash);
    return;
  }
  if (state.draft.tissue === "flush") {
    if (key === "1") chooseInjury("healthy").catch(flash);
    if (key === "2") chooseInjury("mild").catch(flash);
    if (key === "3") chooseInjury("severe").catch(flash);
    if (key === "4") chooseInjury("uncertain").catch(flash);
  }
});

const PREDICT_STEPS = ["sending", "segmenting", "filtering", "scoring_tiles", "aggregating_plants", "ready"];
const PREDICT_TITLES = {
  sending: "Sending photographs to the GPU",
  segmenting: "Finding and cutting out each plant",
  filtering: "Keeping leaf tiles",
  scoring_tiles: "Measuring visible damage",
  aggregating_plants: "Calculating whole-plant severity",
  ready: "Predictions are ready",
  error: "Something went wrong",
};

function paintPredictGpu() {
  const ready = gpuReady();
  const hint = ready
    ? "This computer sends JPGs to the GPU. The GPU cuts tiles, scores visible damage, and sends the report back."
    : "GPU password is not set on this computer. Ask for a local settings file.";
  const tiledHint = ready
    ? "Use this only when the folder already has tiles_foliage.csv and foliage tiles. Raw rover photos belong on the left."
    : "GPU password is not set on this computer. Ask for a local settings file.";
  for (const id of ["form-predict-gpu", "form-predict-tiled"]) {
    const card = $(id);
    if (card) card.classList.toggle("is-off", !ready);
  }
  for (const id of [
    "predict-gpu-path",
    "predict-gpu-path-btn",
    "predict-file-input",
    "predict-folder-input",
    "predict-tiled-path",
    "predict-tiled-path-btn",
    "predict-tiled-folder-input",
  ]) {
    if ($(id)) $(id).disabled = !ready;
  }
  if ($("predict-gpu-hint")) $("predict-gpu-hint").textContent = hint;
  if ($("predict-tiled-hint")) $("predict-tiled-hint").textContent = tiledHint;
}

function predictName() {
  const home = ($("annotator")?.value || "").trim();
  const local = ($("predict-annotator")?.value || "").trim();
  const name = local || home || localStorage.getItem(NAME_KEY) || "";
  if (!name) {
    flash({ message: "Type your name first." });
    return "";
  }
  localStorage.setItem(NAME_KEY, name);
  if ($("annotator")) $("annotator").value = name;
  if ($("predict-annotator")) $("predict-annotator").value = name;
  return name;
}

function predictRunName() {
  return ($("predict-run-name")?.value || "").trim();
}

function setPredictProgress(step, title, detail) {
  $("predict-progress-step").textContent = step;
  $("predict-progress-title").textContent = title;
  $("predict-progress-detail").textContent = detail || "";
  for (const item of document.querySelectorAll("#predict-steps li")) {
    item.classList.toggle("active", item.dataset.step === step);
    item.classList.toggle("done", PREDICT_STEPS.indexOf(item.dataset.step) < PREDICT_STEPS.indexOf(step));
  }
}

function showPredictPanel(panel) {
  if ($("predict-setup")) $("predict-setup").hidden = panel !== "upload";
  $("predict-upload").hidden = panel !== "upload";
  $("predict-progress").hidden = panel !== "progress";
  $("predict-results").hidden = panel !== "results";
  $("predict-detail").hidden = panel !== "detail";
  if (panel !== "upload") $("predict-runs").hidden = true;
}

function stopPredictionPoll() {
  if (state.predictionPoll) {
    clearInterval(state.predictionPoll);
    state.predictionPoll = null;
  }
}

function resetPredictLanding() {
  stopPredictionPoll();
  state.predictionRunId = null;
  state.predictionJobId = null;
  state.predictionRun = null;
  state.plants = [];
  state.selectedPlant = null;
  state.plantDetail = null;
  state.selectedTile = null;
  state.plantView = null;
}

async function initPredictPage() {
  paintPredictGpu();
  if (!state.predictionRunId) {
    showPredictPanel("upload");
    await loadPredictionRuns();
  }
}

async function loadPredictionRuns() {
  const runs = await api("/api/prediction-runs");
  const listed = runs.filter((item) => {
    if (item.status !== "ready" && item.status !== "error") return false;
    if (item.source === "fixture") return false;
    if (String(item.run_id || "").startsWith("_")) return false;
    return true;
  });
  $("predict-run-list").innerHTML = listed
    .map((item) => {
      const meta = item.status === "error"
        ? "Failed"
        : `${item.scored || 0} scored · ${item.failed || 0} failed`;
      return `<li><button type="button" data-run="${item.run_id}">
      <span>${item.name || item.run_id}</span>
      <span class="mono">${meta}</span>
    </button></li>`;
    })
    .join("");
  $("predict-runs").hidden = Boolean($("predict-upload").hidden) || listed.length === 0;
  return listed;
}

function cutoffQuery() {
  return `high_frac=${state.highFrac}&mid_frac=${state.midFrac}`;
}

async function watchPredictionJob(jobId, runId) {
  state.predictionJobId = jobId;
  state.predictionRunId = runId;
  show("predict", true);
  showPredictPanel("progress");
  setPredictProgress("sending", PREDICT_TITLES.sending, "");
  stopPredictionPoll();
  const tick = async () => {
    try {
      const job = await api(`/api/prediction-jobs/${jobId}`);
      setPredictProgress(job.step || "sending", PREDICT_TITLES[job.step] || job.step, job.detail);
      if (job.status === "ready") {
        stopPredictionPoll();
        await openPredictionRun(job.run_id || runId);
      }
      if (job.status === "error") {
        stopPredictionPoll();
        setPredictProgress("error", "Something went wrong", job.detail);
        flash(job.detail);
      }
    } catch (error) {
      const message = error.message || String(error);
      if (/not found/i.test(message)) {
        stopPredictionPoll();
        setPredictProgress("error", "Something went wrong", message);
        flash(error);
      }
    }
  };
  await tick();
  if (state.predictionJobId === jobId) {
    state.predictionPoll = setInterval(tick, 1500);
  }
}

async function openPredictionRun(runId) {
  state.predictionRunId = runId;
  const payload = await api(`/api/prediction-runs/${encodeURIComponent(runId)}/plants?${cutoffQuery()}`);
  state.predictionRun = payload.run;
  state.plants = payload.plants || [];
  if (payload.run?.status === "running") {
    show("predict", true);
    showPredictPanel("progress");
    if (payload.run.job_id) await watchPredictionJob(payload.run.job_id, runId);
    return;
  }
  if (payload.run?.status === "error") {
    show("predict", true);
    showPredictPanel("progress");
    setPredictProgress("error", "Something went wrong", payload.run.error || "");
    return;
  }
  show("predict");
  renderPredictTable();
}

function renderPredictTable() {
  showPredictPanel("results");
  const run = state.predictionRun || {};
  const scored = state.plants.filter((row) => row.status === "scored");
  const failed = state.plants.filter((row) => row.status === "failed");
  $("predict-result-kicker").textContent = run.name || "Ranking";
  $("predict-result-title").textContent = scored.length === 1 ? "Visible damage for this plant" : "Most damaged first";
  $("predict-result-counts").textContent =
    `${scored.length} scored · ${failed.length} failed · high is the top ${Math.round(state.highFrac * 100)}% in this upload`;
  $("predict-export-plants").href = `/api/prediction-runs/${encodeURIComponent(run.run_id)}/export?${cutoffQuery()}`;
  $("predict-export-tiles").href = `/api/prediction-runs/${encodeURIComponent(run.run_id)}/export?level=tiles`;
  const body = $("predict-table").querySelector("tbody");
  body.innerHTML = state.plants
    .map((row) => {
      const failedRow = row.status === "failed";
      const review = row.needs_review || row.low_flush_warning;
      return `<tr data-image="${encodeURIComponent(row.image)}" class="${failedRow ? "is-failed" : ""} ${review ? "is-review" : ""}">
        <td>${row.rank ?? "—"}</td>
        <td>${row.image}</td>
        <td>${failedRow ? "—" : Number(row.predicted_expected).toFixed(2)}</td>
        <td>${failedRow ? "—" : row.predicted_score}</td>
        <td>${row.severity_group || "Could not score"}</td>
        <td>${failedRow ? "—" : `${Math.round((row.confidence || 0) * 100)}%`}</td>
        <td>${row.priority_group ? `<span class="chip ${row.priority_group}">${row.priority_group}</span>` : "—"}</td>
        <td>${review ? "Review" : ""}</td>
      </tr>`;
    })
    .join("");
}

function probRows(prefix, values) {
  return values
    .map(([label, value]) => {
      const pct = Math.round((Number(value) || 0) * 100);
      return `<div class="prob-row"><span>${label}</span><div class="prob-bar"><span style="width:${pct}%"></span></div><span>${pct}%</span></div>`;
    })
    .join("");
}

async function openPredictPlant(image) {
  const runId = state.predictionRunId;
  const payload = await api(
    `/api/prediction-runs/${encodeURIComponent(runId)}/plants/${encodeURIComponent(image)}?${cutoffQuery()}`
  );
  state.plantDetail = payload;
  state.selectedPlant = image;
  state.selectedTile = null;
  showPredictPanel("detail");
  persist(true);
  const plant = payload.plant;
  $("predict-detail-kicker").textContent = plant.status === "failed" ? "Could not score" : "Plant evidence";
  $("predict-detail-title").textContent = image;
  $("predict-detail-summary").textContent = plant.status === "failed"
    ? `${plant.failure === "no_plant_detected" ? "No plant was found." : "Not enough leaf tiles were kept."} This is not a healthy score.`
    : `Damage index ${Number(plant.predicted_expected).toFixed(2)} · rounded ${plant.predicted_score} · ${plant.severity_group} · ${plant.tile_count} tiles · ${plant.priority_group || "unranked"} priority in this upload`;
  const warning = plant.warning || (plant.needs_review ? "Low confidence. Review this plant before using the rank." : "");
  $("predict-detail-warning").hidden = !warning;
  $("predict-detail-warning").textContent = warning;
  const hasTiles = plant.status === "scored" && (payload.tiles || []).some((tile) => tile.tile);
  const hasCrop = Boolean(payload.media?.has_crop && payload.media.crop);
  const evidence = $("predict-evidence");
  if (evidence) {
    evidence.classList.toggle("no-original", !payload.media?.has_original);
    evidence.classList.toggle("no-crop", !hasCrop);
  }
  const originalStage = $("predict-original-stage");
  const original = $("predict-original");
  if (payload.media?.has_original && payload.media.original) {
    if (originalStage) originalStage.hidden = false;
    original.onerror = () => {
      original.removeAttribute("src");
      if (originalStage) originalStage.hidden = true;
      if (evidence) evidence.classList.add("no-original");
    };
    original.src = payload.media.original;
  } else {
    original.removeAttribute("src");
    if (originalStage) originalStage.hidden = true;
  }
  const overlayStage = $("predict-overlay-stage");
  if (overlayStage) overlayStage.hidden = !hasCrop;
  const crop = $("predict-crop");
  if (crop) {
    crop.hidden = true;
    crop.removeAttribute("src");
  }
  state.plantView = null;
  if (hasCrop) {
    buildPlantView().catch(flash);
  }
  $("predict-scores").innerHTML = plant.status === "failed"
    ? `<p>No 1 to 5 score was written for this plant.</p><p>${plant.tile_count || 0} leaf tiles kept.</p>`
    : `<p><strong>${Number(plant.predicted_expected).toFixed(2)}</strong> damage index</p>
       <p>Rounded score ${plant.predicted_score}</p>
       <p>Confidence ${Math.round((plant.confidence || 0) * 100)}%</p>
       ${probRows("score", [1, 2, 3, 4, 5].map((score) => [`${score}`, plant[`p_score_${score}`]]))}`;
  const evidenceTiles = payload.evidence || [];
  if ($("predict-evidence-block")) $("predict-evidence-block").hidden = evidenceTiles.length === 0;
  $("predict-evidence-tiles").innerHTML = evidenceTiles
    .map((tile) => `<button type="button" data-tile="${encodeURIComponent(tile.tile)}" title="${tile.tile}">
      <img src="${tile.url || `/media/predictions/${encodeURIComponent(runId)}/tile/${encodeURIComponent(tile.tile)}`}" alt="${tile.tile}">
    </button>`)
    .join("");
  $("predict-tile-prob").hidden = true;
  requestAnimationFrame(drawPredictOverlay);
}

function cropBox() {
  const info = state.plantDetail?.image || {};
  return {
    x0: Number(info.box_x0 || 0),
    y0: Number(info.box_y0 || 0),
    w: Number(info.box_w || info.crop_width || info.width || 0),
    h: Number(info.box_h || info.crop_height || info.height || 0),
  };
}

function tileOnCrop(tile) {
  if (tile.crop_w != null && tile.crop_h != null) {
    return {
      x: Number(tile.crop_x),
      y: Number(tile.crop_y),
      w: Number(tile.crop_w),
      h: Number(tile.crop_h),
    };
  }
  const box = cropBox();
  const srcW = Math.max(box.w, 1);
  const srcH = Math.max(box.h, 1);
  const x = Number(tile.x || 0) - box.x0;
  const y = Number(tile.y || 0) - box.y0;
  const w = Number(tile.width || 512);
  const h = Number(tile.height || 512);
  const left = Math.max(0, x);
  const top = Math.max(0, y);
  const right = Math.min(srcW, x + w);
  const bottom = Math.min(srcH, y + h);
  if (right - left < 1 || bottom - top < 1) return null;
  return { x: left, y: top, w: right - left, h: bottom - top };
}

function damageColor(healthy, mild, severe) {
  const expected = mild + 2 * severe;
  if (expected >= 1.35) return [220, 20, 20];
  if (expected >= 0.70) return [240, 180, 0];
  return [20, 150, 48];
}

function tileMediaUrl(tile) {
  if (tile.url) return tile.url;
  const runId = state.predictionRunId;
  return `/media/predictions/${encodeURIComponent(runId)}/tile/${encodeURIComponent(tile.tile)}`;
}

function loadTileImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load ${url}`));
    image.src = url;
  });
}

function setOverlayStatus(text) {
  const node = $("predict-overlay-status");
  if (!node) return;
  node.hidden = !text;
  node.textContent = text || "";
}

async function buildPlantView() {
  const recon = $("predict-reconstruct");
  const overlay = $("predict-overlay");
  if (!recon || !overlay || !state.plantDetail) return;
  const token = `${state.predictionRunId}:${state.selectedPlant}`;
  const cropUrl = state.plantDetail.media?.crop;
  if (!cropUrl) {
    setOverlayStatus("No plant photograph is available for this plant.");
    state.plantView = null;
    return;
  }
  setOverlayStatus("");
  let cropImage;
  try {
    cropImage = await loadTileImage(cropUrl);
  } catch {
    setOverlayStatus("The plant photograph could not be loaded.");
    state.plantView = null;
    return;
  }
  if (`${state.predictionRunId}:${state.selectedPlant}` !== token) return;
  const cropW = cropImage.naturalWidth || cropImage.width;
  const cropH = cropImage.naturalHeight || cropImage.height;
  const scale = Math.min(1, 900 / Math.max(cropW, cropH));
  const width = Math.max(1, Math.round(cropW * scale));
  const height = Math.max(1, Math.round(cropH * scale));
  recon.width = width;
  recon.height = height;
  overlay.width = width;
  overlay.height = height;
  recon.getContext("2d").drawImage(cropImage, 0, 0, width, height);
  state.plantView = {
    width,
    height,
    x0: 0,
    y0: 0,
    cropW,
    cropH,
    token,
  };
  paintPlantHeatmap();
}

function paintPlantTileRails(ctx, view) {
  const scaleX = view.width / view.cropW;
  const scaleY = view.height / view.cropH;
  const rail = Math.max(5, Math.round(view.width / 90));
  for (const tile of state.plantDetail?.tiles || []) {
    const flush = Number(tile.p_flush || 0);
    if (flush < 0.22) continue;
    const rect = tileOnCrop(tile);
    if (!rect) continue;
    const x = (rect.x - view.x0) * scaleX;
    const y = (rect.y - view.y0) * scaleY;
    const w = rect.w * scaleX;
    const h = rect.h * scaleY;
    const [r, g, b] = damageColor(
      Number(tile.p_healthy || 0),
      Number(tile.p_mild || 0),
      Number(tile.p_severe || 0)
    );
    const color = `rgb(${r}, ${g}, ${b})`;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.1;
    ctx.fillRect(x, y, w, h);
    ctx.globalAlpha = 0.92;
    ctx.fillRect(x, y, rail, h);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.lineWidth = state.selectedTile === tile.tile ? Math.max(4, view.width / 160) : Math.max(2, view.width / 260);
    ctx.strokeRect(x + 1, y + 1, w - 2, h - 2);
  }
}

function paintPlantHeatmap() {
  const overlay = $("predict-overlay");
  const view = state.plantView;
  if (!overlay || !view) return;
  overlay.width = view.width;
  overlay.height = view.height;
  const ctx = overlay.getContext("2d");
  ctx.clearRect(0, 0, view.width, view.height);
  paintPlantTileRails(ctx, view);
}

function drawPredictOverlay() {
  if (state.plantView) paintPlantHeatmap();
}

function tileAtOverlay(event) {
  const canvas = $("predict-overlay");
  const view = state.plantView;
  if (!canvas || !view) return null;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return null;
  const x = view.x0 + (event.clientX - bounds.left) * (view.cropW / bounds.width);
  const y = view.y0 + (event.clientY - bounds.top) * (view.cropH / bounds.height);
  let best = null;
  let bestFlush = -1;
  for (const tile of state.plantDetail?.tiles || []) {
    const rect = tileOnCrop(tile);
    if (!rect) continue;
    if (x >= rect.x && x <= rect.x + rect.w && y >= rect.y && y <= rect.y + rect.h) {
      const flush = Number(tile.p_flush || 0);
      if (flush >= bestFlush) {
        best = tile;
        bestFlush = flush;
      }
    }
  }
  return best;
}

function showTileProb(tile) {
  if (!tile) {
    $("predict-tile-prob").hidden = true;
    return;
  }
  state.selectedTile = tile.tile;
  $("predict-tile-prob").hidden = false;
  $("predict-tile-prob").innerHTML =
    `<p>${tile.tile}</p>` +
    probRows("tile", [
      ["New growth", tile.p_flush],
      ["Healthy", tile.p_healthy],
      ["Mild", tile.p_mild],
      ["Severe", tile.p_severe],
    ]);
  for (const button of document.querySelectorAll("#predict-evidence-tiles [data-tile]")) {
    button.classList.toggle("active", decodeURIComponent(button.dataset.tile) === tile.tile);
  }
  drawPredictOverlay();
}

async function startPredictFromPath() {
  const name = predictName();
  if (!name) return;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return;
  }
  const path = $("predict-gpu-path").value.trim();
  if (!path) {
    flash({ message: "Choose photographs or paste a folder path." });
    return;
  }
  const job = await api("/api/prediction-jobs/gpu", {
    method: "POST",
    body: JSON.stringify({ annotator: name, name: predictRunName(), path }),
  });
  await watchPredictionJob(job.job_id, job.run_id);
}

async function startPredictFromTiled() {
  const name = predictName();
  if (!name) return;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return;
  }
  const path = $("predict-tiled-path").value.trim();
  if (!path) {
    flash({ message: "Paste the path to a tiled run." });
    return;
  }
  const job = await api("/api/prediction-jobs/tiled", {
    method: "POST",
    body: JSON.stringify({ annotator: name, name: predictRunName(), path }),
  });
  await watchPredictionJob(job.job_id, job.run_id);
}

function isTiledFileList(fileList) {
  return [...fileList].some((file) => {
    const name = (file.webkitRelativePath || file.name).split("/").pop();
    return name === "tiles_foliage.csv";
  });
}

async function postPredictionUpload(url, fileList, emptyMessage) {
  const name = predictName();
  if (!name) return null;
  if (!gpuReady()) {
    flash({ message: "GPU password is not set on this computer." });
    return null;
  }
  const files = [...fileList];
  if (!files.length) {
    flash({ message: emptyMessage });
    return null;
  }
  const data = new FormData();
  data.append("annotator", name);
  data.append("name", predictRunName() || files[0].webkitRelativePath.split("/")[0] || files[0].name);
  for (const file of files) {
    data.append("files", file);
    data.append("relpaths", file.webkitRelativePath || file.name);
  }
  show("predict");
  showPredictPanel("progress");
  setPredictProgress("sending", PREDICT_TITLES.sending, `${files.length} files`);
  const response = await fetch(url, { method: "POST", body: data });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

async function startPredictFromFiles(fileList) {
  if (isTiledFileList(fileList)) {
    await startPredictFromTiledFiles(fileList);
    return;
  }
  const job = await postPredictionUpload(
    "/api/prediction-jobs/gpu-upload",
    fileList,
    "Choose photographs or a folder of photographs."
  );
  if (job) await watchPredictionJob(job.job_id, job.run_id);
}

async function startPredictFromTiledFiles(fileList) {
  const job = await postPredictionUpload(
    "/api/prediction-jobs/tiled-upload",
    fileList,
    "Choose a tiled folder with tiles_foliage.csv."
  );
  if (job) await watchPredictionJob(job.job_id, job.run_id);
}

function bindPredictDrop(zoneId, inputId, handler) {
  const zone = $(zoneId);
  const input = $(inputId);
  if (!zone || !input) return;
  zone.addEventListener("dragover", (event) => {
    event.preventDefault();
    zone.classList.add("is-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("is-over"));
  zone.addEventListener("drop", async (event) => {
    event.preventDefault();
    zone.classList.remove("is-over");
    try {
      const files = await filesFromDrop(event);
      await handler(files);
    } catch (error) {
      flash(error);
      showPredictPanel("upload");
    }
  });
  input.addEventListener("change", async (event) => {
    try {
      await handler(event.target.files);
    } catch (error) {
      flash(error);
      showPredictPanel("upload");
    }
  });
}

$("form-predict-gpu")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await startPredictFromPath();
  } catch (error) {
    flash(error);
    showPredictPanel("upload");
  }
});

$("form-predict-tiled")?.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await startPredictFromTiled();
  } catch (error) {
    flash(error);
    showPredictPanel("upload");
  }
});

bindPredictDrop("predict-file-drop", "predict-file-input", startPredictFromFiles);
bindPredictDrop("predict-folder-drop", "predict-folder-input", startPredictFromFiles);
bindPredictDrop("predict-tiled-drop", "predict-tiled-folder-input", startPredictFromTiledFiles);

$("predict-run-list")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-run]");
  if (button) openPredictionRun(button.dataset.run).catch(flash);
});

$("predict-table")?.addEventListener("click", (event) => {
  const row = event.target.closest("[data-image]");
  if (!row) return;
  openPredictPlant(decodeURIComponent(row.dataset.image)).catch(flash);
});

$("predict-new")?.addEventListener("click", () => {
  resetPredictLanding();
  show("predict");
  initPredictPage().catch(flash);
});

$("predict-back")?.addEventListener("click", () => {
  state.selectedPlant = null;
  renderPredictTable();
  persist(true);
});

$("predict-overlay")?.addEventListener("click", (event) => {
  showTileProb(tileAtOverlay(event));
});

$("predict-evidence-tiles")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-tile]");
  if (!button || !state.plantDetail) return;
  const name = decodeURIComponent(button.dataset.tile);
  const tile = state.plantDetail.tiles.find((item) => item.tile === name)
    || state.plantDetail.evidence.find((item) => item.tile === name);
  showTileProb(tile);
});

async function refreshPredictionCutoffs() {
  if (!state.predictionRunId) return;
  const plant = state.selectedPlant;
  await openPredictionRun(state.predictionRunId);
  if (plant) await openPredictPlant(plant);
}

$("predict-high-frac")?.addEventListener("change", () => {
  state.highFrac = Math.max(0.05, Math.min(0.9, Number($("predict-high-frac").value) / 100));
  refreshPredictionCutoffs().catch(flash);
});

$("predict-mid-frac")?.addEventListener("change", () => {
  state.midFrac = Math.max(0.05, Math.min(0.9, Number($("predict-mid-frac").value) / 100));
  refreshPredictionCutoffs().catch(flash);
});

window.addEventListener("resize", () => {
  if (state.screen === "predict" && state.plantDetail) drawPredictOverlay();
});

window.addEventListener("popstate", () => {
  boot({ fromHistory: true }).catch(flash);
});

async function boot() {
  let sessions = [];
  try {
    await loadSettings();
    sessions = await loadSessions();
  } catch (error) {
    flash(error);
  }
  const located = parseLocation();
  if (located?.screen === "guide") {
    show("guide", true);
    const hash = location.hash;
    if (hash) {
      requestAnimationFrame(() => document.querySelector(hash)?.scrollIntoView());
    }
    return;
  }
  if (located?.screen === "predict") {
    show("predict", true);
    paintPredictGpu();
    if (located.runId) {
      await openPredictionRun(located.runId);
      if (located.plant) await openPredictPlant(located.plant);
    } else {
      await initPredictPage();
    }
    return;
  }
  const known = sessions.find((item) => item.batch_id === located?.batchId);
  if (located && known) {
    const session = readSession();
    state.plantFilter = located.plantFilter || session?.plantFilter || "";
    if (located.screen === "review") {
      state.fromReview = false;
      state.batch = await api(`/api/batches/${located.batchId}`);
      show("review", true);
      await renderReview();
      return;
    }
    await openBatch(located.batchId, located.tileId, {
      replace: true,
      fromReview: Boolean(session?.fromReview),
    });
    return;
  }
  show("load", true);
}

boot().catch(flash);
