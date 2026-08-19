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
  draft: { tissue: null, injury: null, curl: null },
  fromReview: false,
  plantFilter: "",
  skipPush: false,
};

const TISSUE_NAME = {
  flush: "new growth",
  mature: "old leaves",
  tube: "not a leaf",
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
  if (!state.batch) return;
  const session = {
    batchId: state.batch.id,
    tileId: state.tile?.id || null,
    screen: state.screen,
    plantFilter: state.plantFilter || "",
    fromReview: state.fromReview,
  };
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
  const url = pathFor();
  const next = `${location.origin}${url}`;
  if (next === location.href) return;
  const method = replace || state.skipPush ? "replaceState" : "pushState";
  history[method](session, "", url);
}

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
  } catch {
    return null;
  }
}

function parseLocation() {
  const match = location.pathname.match(/^\/batches\/(\d+)\/(label|review)$/);
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
  for (const id of ["load", "progress", "label", "review"]) {
    const node = $(`screen-${id}`);
    const visible = id === screen;
    node.hidden = !visible;
    node.classList.toggle("is-visible", visible);
  }
  $("top-nav").hidden = !state.batch || screen === "progress";
  for (const button of document.querySelectorAll("#top-nav [data-go]")) {
    button.classList.toggle("active", button.dataset.go === screen);
  }
  if (state.batch) {
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
    curl: fromLabel?.curl || null,
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
  for (const button of document.querySelectorAll("[data-curl]")) {
    button.classList.toggle("active", flush && button.dataset.curl === state.draft.curl);
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
    <span>${counts.labeled} labeled · ${counts.unlabeled} left · ${counts.flush_injured || 0} new growth injured</span>
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
      <button type="button" class="${item.id === state.tile.id ? "current" : ""} ${tileTone(item)}" data-tile="${item.id}" title="${TISSUE_NAME[item.tissue] || "unlabeled"} ${item.curl ? `curl ${item.curl}` : ""}">
        <img src="/media/tile/${item.id}" alt="${item.tile}">
      </button>`
    )
    .join("");
  const currentThumb = $("filmstrip").querySelector(".current");
  if (currentThumb) currentThumb.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
}

function draftStatus() {
  if (!state.draft.tissue) return "Mark the tissue first.";
  if (state.draft.tissue !== "flush") return `${TISSUE_NAME[state.draft.tissue]} saved as skip.`;
  if (!state.draft.injury) return "New growth. Now mark healthy, injured, or skip.";
  if (state.draft.injury === "skip") return "New growth skip. Not scored.";
  if (!state.draft.curl) return `New growth ${state.draft.injury}. Now mark curl yes or no.`;
  return `New growth ${state.draft.injury}, curl ${state.draft.curl}.`;
}

function canSaveDraft() {
  if (!state.draft.tissue) return false;
  if (state.draft.tissue !== "flush") return true;
  if (state.draft.injury === "skip") return true;
  return Boolean(state.draft.injury && state.draft.curl);
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
      curl: state.draft.curl,
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
    state.draft.curl = null;
    paintDraft();
    await saveDraft();
    return;
  }
  state.draft.injury = null;
  state.draft.curl = null;
  paintDraft();
  stashDraft();
}

async function chooseInjury(injury) {
  if (!state.tile || state.draft.tissue !== "flush") return;
  state.draft.injury = injury;
  if (injury === "skip") {
    state.draft.curl = null;
    paintDraft();
    await saveDraft();
    return;
  }
  paintDraft();
  stashDraft();
  if (state.draft.curl) await saveDraft();
}

async function chooseCurl(curl) {
  if (!state.tile || state.draft.tissue !== "flush") return;
  if (!state.draft.injury || state.draft.injury === "skip") {
    flash({ message: "Mark healthy or injured first." });
    return;
  }
  state.draft.curl = curl;
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
    `${counts.flush_healthy || 0} new growth healthy · ${counts.flush_injured || 0} new growth injured · ${counts.curl_yes || 0} curl · ${counts.mature || 0} old leaves · ${counts.unlabeled} left`;
  const select = $("review-plant");
  select.innerHTML =
    `<option value="">All plants</option>` +
    (state.batch.images || [])
      .map((item) => `<option value="${item.id}">${item.filename}</option>`)
      .join("");
  select.value = state.plantFilter;
  for (const name of ["flush_healthy", "flush_injured", "mature", "skip"]) {
    const column = document.querySelector(`[data-col="${name}"]`);
    const tiles = payload.tiles.filter((item) => reviewBucket(item) === name);
    column.innerHTML = tiles
      .map((item) => {
        const curl = item.curl
          ? `<span class="curl-chip">${item.curl === "yes" ? "curl" : "flat"}</span>`
          : "";
        return `<button type="button" data-tile="${item.id}" title="${item.image} ${item.tissue || ""} ${item.injury || ""} ${item.curl ? `curl ${item.curl}` : ""}"><img src="/media/tile/${item.id}" alt="${item.tile}">${curl}</button>`;
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
    exportGpu.hidden = !ready || !state.batch || state.screen === "progress" || state.screen === "load";
    exportGpu.disabled = !ready || !state.batch;
  }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  $("annotator").value = localStorage.getItem(NAME_KEY) || "";
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
    const reader = entry.createReader();
    const entries = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    for (const child of entries) {
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
  const go = event.target.dataset.go;
  if (go === "load") {
    state.fromReview = false;
    show("load");
    loadSessions().catch(flash);
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

document.querySelector(".columns").addEventListener("click", (event) => {
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
  const curl = event.target.closest("[data-curl]");
  if (injury) chooseInjury(injury.dataset.injury).catch(flash);
  if (curl) chooseCurl(curl.dataset.curl).catch(flash);
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
  if (state.draft.tissue === "flush" && !state.draft.injury) {
    if (key === "1") chooseInjury("healthy").catch(flash);
    if (key === "2") chooseInjury("injured").catch(flash);
    if (key === "3") chooseInjury("skip").catch(flash);
    return;
  }
  if (key === "y" || key === "Y") chooseCurl("yes").catch(flash);
  if (key === "n" || key === "N") chooseCurl("no").catch(flash);
});

window.addEventListener("popstate", () => {
  boot({ fromHistory: true }).catch(flash);
});

async function boot() {
  await loadSettings();
  const sessions = await loadSessions();
  const located = parseLocation();
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
