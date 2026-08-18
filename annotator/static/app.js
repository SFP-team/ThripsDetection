const state = {
  screen: "load",
  settings: null,
  batch: null,
  tile: null,
  plant: [],
  jobId: null,
  poll: null,
  draft: { tissue: null, injury: null, curl: null },
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

function show(screen) {
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
}

function annotator() {
  return $("annotator").value.trim() || state.settings?.annotator || "lab";
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

function resetDraft(fromLabel) {
  state.draft = {
    tissue: fromLabel?.tissue || null,
    injury: fromLabel?.injury || fromLabel?.label || null,
    curl: fromLabel?.curl || null,
  };
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
    const order = ["finding", "cutting", "leaves", "copying", "ready"];
    item.classList.toggle("done", order.indexOf(item.dataset.step) < order.indexOf(step));
  }
}

const STEP_TITLES = {
  finding: "Finding the plant",
  cutting: "Cutting it out of the background",
  leaves: "Keeping only leaf squares",
  copying: "Copying tiles",
  ready: "Ready to label",
  error: "Something went wrong",
};

async function watchJob(jobId) {
  state.jobId = jobId;
  show("progress");
  if (state.poll) clearInterval(state.poll);
  const tick = async () => {
    const job = await api(`/api/jobs/${jobId}`);
    setProgress(job.step || "finding", STEP_TITLES[job.step] || job.step, job.detail);
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

async function openBatch(batchId, tileId = null) {
  state.batch = await api(`/api/batches/${batchId}`);
  const query = tileId ? `tile_id=${tileId}` : "";
  const payload = await api(`/api/batches/${batchId}/next?${query}`);
  state.batch = payload.batch;
  if (payload.done && !tileId) {
    state.tile = null;
    state.plant = [];
    show("label");
    renderDone();
    return;
  }
  state.tile = payload.tile;
  state.plant = payload.plant;
  resetDraft(payload.tile?.current_label);
  show("label");
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
    <span>${counts.labeled} labeled · ${counts.unlabeled} left · ${counts.flush_injured || 0} flush injured</span>
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
      <button type="button" class="${item.id === state.tile.id ? "current" : ""} ${tileTone(item)}" data-tile="${item.id}" title="${item.tissue || "unlabeled"} ${item.curl ? `curl ${item.curl}` : ""}">
        <img src="/media/tile/${item.id}" alt="${item.tile}">
      </button>`
    )
    .join("");
  const currentThumb = $("filmstrip").querySelector(".current");
  if (currentThumb) currentThumb.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
}

function draftStatus() {
  if (!state.draft.tissue) return "Mark the tissue first.";
  if (state.draft.tissue !== "flush") return `${state.draft.tissue} — saved as skip.`;
  if (!state.draft.injury) return "Flush — now mark healthy, injured, or skip.";
  if (state.draft.injury === "skip") return "Flush skip — not scored.";
  if (!state.draft.curl) return `Flush ${state.draft.injury} — now mark curl yes or no.`;
  return `Flush ${state.draft.injury}, curl ${state.draft.curl}.`;
}

function canSaveDraft() {
  if (!state.draft.tissue) return false;
  if (state.draft.tissue !== "flush") return true;
  if (state.draft.injury === "skip") return true;
  return Boolean(state.draft.injury && state.draft.curl);
}

async function saveDraft() {
  if (!state.tile || !canSaveDraft()) return;
  const payload = await api("/api/label", {
    method: "POST",
    body: JSON.stringify({
      tile_id: state.tile.id,
      tissue: state.draft.tissue,
      injury: state.draft.injury,
      curl: state.draft.curl,
      annotator: annotator(),
    }),
  });
  state.batch = payload.batch;
  if (payload.done) {
    state.tile = null;
    state.plant = [];
    renderDone();
    return;
  }
  state.tile = payload.next;
  state.plant = payload.plant;
  resetDraft(null);
  renderLabel();
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
  resetDraft(payload.tile?.current_label);
  show("label");
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
  const imageId = $("review-plant").value;
  const query = imageId ? `image_id=${imageId}` : "";
  const payload = await api(`/api/batches/${state.batch.id}/review?${query}`);
  state.batch = payload.batch;
  const counts = state.batch.counts;
  $("review-counts").textContent =
    `${counts.flush_healthy || 0} flush healthy · ${counts.flush_injured || 0} flush injured · ${counts.curl_yes || 0} curl · ${counts.mature || 0} mature · ${counts.unlabeled} left`;
  const select = $("review-plant");
  const current = select.value;
  select.innerHTML =
    `<option value="">All plants</option>` +
    (state.batch.images || [])
      .map((item) => `<option value="${item.id}">${item.filename}</option>`)
      .join("");
  select.value = current;
  for (const name of ["flush_healthy", "flush_injured", "mature", "skip"]) {
    const column = document.querySelector(`[data-col="${name}"]`);
    const tiles = payload.tiles.filter((item) => reviewBucket(item) === name);
    column.innerHTML = tiles
      .map(
        (item) => `
        <button type="button" data-tile="${item.id}" title="${item.image} ${item.tissue || ""} ${item.injury || ""} ${item.curl ? `curl ${item.curl}` : ""}">
          <img src="/media/tile/${item.id}" alt="${item.tile}">
          ${item.curl ? `<span class="curl-chip">${item.curl === "yes" ? "curl" : "flat"}</span>` : ""}
        </button>`
      )
      .join("");
  }
}

async function loadSettings() {
  state.settings = await api("/api/settings");
  $("annotator").value = state.settings.annotator || "lab";
  for (const key of [
    "pipeline_mode",
    "ssh_host",
    "ssh_user",
    "remote_existing_run",
    "local_existing_run",
    "local_project",
  ]) {
    if ($(key)) $(key).value = state.settings[key] || "";
  }
  if (state.settings.ssh_password_set) {
    $("ssh_password").placeholder = "saved";
  }
}

async function loadBatches() {
  const batches = await api("/api/batches");
  const ready = batches.filter((item) => item.status === "ready");
  $("batch-list").hidden = ready.length === 0;
  $("batches").innerHTML = ready
    .map((item) => {
      const left = item.counts.unlabeled;
      return `<li><button type="button" data-batch="${item.id}">
        <span>${item.name}</span>
        <span class="mono">BAT-${item.id} · ${item.counts.tiles} tiles · ${left} left</span>
      </button></li>`;
    })
    .join("");
}

$("form-import").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const path = $("import-path").value.trim();
    const job = await api("/api/import", {
      method: "POST",
      body: JSON.stringify({
        annotator: annotator(),
        path,
        remote: !path,
        name: path ? "Local foliage tiles" : "Existing foliage tiles",
      }),
    });
    await watchJob(job.job_id);
  } catch (error) {
    flash(error);
  }
});

$("form-prepare").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const job = await api("/api/prepare", {
      method: "POST",
      body: JSON.stringify({
        annotator: annotator(),
        path: $("prepare-path").value.trim(),
        name: $("prepare-name").value.trim(),
      }),
    });
    await watchJob(job.job_id);
  } catch (error) {
    flash(error);
  }
});

$("form-settings").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const body = {
      annotator: annotator(),
      pipeline_mode: $("pipeline_mode").value,
      ssh_host: $("ssh_host").value,
      ssh_user: $("ssh_user").value,
      remote_existing_run: $("remote_existing_run").value,
      local_existing_run: $("local_existing_run").value,
      local_project: $("local_project").value,
    };
    const password = $("ssh_password").value;
    if (password) body.ssh_password = password;
    state.settings = await api("/api/settings", {
      method: "POST",
      body: JSON.stringify(body),
    });
    $("ssh_password").value = "";
    $("ssh_password").placeholder = state.settings.ssh_password_set ? "saved" : "unchanged if blank";
    flash({ message: "Settings saved" });
  } catch (error) {
    flash(error);
  }
});

$("file-input").addEventListener("change", uploadFiles);
$("drop-zone").addEventListener("dragover", (event) => {
  event.preventDefault();
  $("drop-zone").classList.add("is-over");
});
$("drop-zone").addEventListener("dragleave", () => {
  $("drop-zone").classList.remove("is-over");
});
$("drop-zone").addEventListener("drop", (event) => {
  event.preventDefault();
  $("drop-zone").classList.remove("is-over");
  uploadFiles({ target: { files: event.dataTransfer.files } });
});

async function uploadFiles(event) {
  const files = [...(event.target.files || [])];
  if (!files.length) return;
  const data = new FormData();
  data.append("annotator", annotator());
  data.append("name", $("prepare-name").value.trim() || "Uploaded photos");
  for (const file of files) data.append("files", file);
  try {
    show("progress");
    setProgress("finding", "Uploading photos", `${files.length} files`);
    const response = await fetch("/api/prepare/upload", { method: "POST", body: data });
    if (!response.ok) throw new Error(await response.text());
    const job = await response.json();
    await watchJob(job.job_id);
  } catch (error) {
    flash(error);
    show("load");
  }
}

$("batches").addEventListener("click", (event) => {
  const button = event.target.closest("[data-batch]");
  if (button) openBatch(Number(button.dataset.batch));
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

$("filmstrip").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tile]");
  if (button) openBatch(state.batch.id, Number(button.dataset.tile)).catch(flash);
});

$("top-nav").addEventListener("click", (event) => {
  const go = event.target.dataset.go;
  if (go === "load") show("load");
  if (go === "label" && state.batch) openBatch(state.batch.id).catch(flash);
  if (go === "review" && state.batch) {
    show("review");
    renderReview().catch(flash);
  }
});

$("review-plant").addEventListener("change", () => renderReview().catch(flash));

document.querySelector(".columns").addEventListener("click", (event) => {
  const button = event.target.closest("[data-tile]");
  if (!button) return;
  openBatch(state.batch.id, Number(button.dataset.tile)).catch(flash);
});

document.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName);
  if (typing || state.screen !== "label") return;
  const key = event.key;
  if (key === "Escape") {
    resetDraft(state.tile?.current_label);
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
    if (next) openBatch(state.batch.id, next.id).catch(flash);
    return;
  }
  if (key === "ArrowLeft" && state.plant.length && state.tile) {
    const index = state.plant.findIndex((item) => item.id === state.tile.id);
    const prev = state.plant[index - 1];
    if (prev) openBatch(state.batch.id, prev.id).catch(flash);
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

loadSettings().then(loadBatches).catch(flash);
