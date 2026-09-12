from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

from annotator.config import DATA, ROOT, load_settings, public_settings
from annotator.db import (
    add_label,
    get_batch,
    get_tile,
    init_db,
    list_batches,
    media_path,
    next_unlabeled,
    plant_tiles,
    review_tiles,
    set_batch_annotator,
    undo_label,
)
from annotator.export_merge import EXPORT_FIELDS
from annotator.gpu_jobs import export_batch_to_gpu, start_gpu_session, start_gpu_session_from_upload
from annotator.pipeline import get_job
from annotator.predict_jobs import (
    compare_run_to_deployment,
    export_plants,
    export_tiles,
    get_prediction_job,
    iter_remote_file,
    list_prediction_runs,
    media_content_type,
    plant_detail,
    read_run,
    resolve_prediction_media,
    resume_running_prediction_jobs,
    start_final_test_repro,
    start_prediction_from_path,
    start_prediction_from_remote_path,
    start_prediction_from_tiled,
    start_prediction_from_tiled_upload,
    start_prediction_from_upload,
)
from annotator.predict_report import EXPORT_FIELDS as PREDICT_EXPORT_FIELDS
from annotator.predict_report import TILE_EXPORT_FIELDS
from annotator.sessions import (
    ensure_legacy_sessions,
    list_sessions,
    start_session_from_dir,
    start_session_from_upload,
    write_local_export,
)

STATIC = ROOT / "static"

app = FastAPI(title="Lab tile annotator")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

KIT_ONLY = "This copy only labels tiles already in the folder."


class LabelIn(BaseModel):
    tile_id: int
    tissue: str
    injury: str | None = None
    curl: str | None = None
    label: str | None = None
    annotator: str = "lab"


class UndoIn(BaseModel):
    batch_id: int


class AnnotatorIn(BaseModel):
    annotator: str = Field(min_length=1)


class SessionPathIn(BaseModel):
    annotator: str = Field(min_length=1)
    name: str = ""
    path: str = Field(min_length=1)


class SessionGpuIn(BaseModel):
    annotator: str = Field(min_length=1)
    name: str = ""
    path: str = Field(min_length=1)


class PredictionPathIn(BaseModel):
    annotator: str = Field(min_length=1)
    name: str = ""
    path: str = Field(min_length=1)


def _require_gpu() -> None:
    settings = load_settings()
    if not settings.ssh_host or not settings.ssh_user or not settings.ssh_password:
        raise HTTPException(400, "GPU password is missing. Add settings.json or .env on this computer.")


@app.on_event("startup")
def startup() -> None:
    init_db()
    DATA.mkdir(parents=True, exist_ok=True)
    ensure_legacy_sessions()
    resume_running_prediction_jobs()


def _index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/")
def index() -> FileResponse:
    return _index()


@app.get("/how-to-annotate")
def how_to_annotate() -> FileResponse:
    return _index()


@app.get("/predict")
@app.get("/predict/{run_id}")
def spa_predict(run_id: str | None = None) -> FileResponse:
    return _index()


@app.get("/batches/{batch_id}/label")
@app.get("/batches/{batch_id}/review")
def spa_batch(batch_id: int) -> FileResponse:
    return _index()


@app.get("/api/settings")
def api_settings() -> dict[str, Any]:
    return public_settings(load_settings())


@app.post("/api/settings")
def api_save_settings() -> dict[str, Any]:
    """Kept for older clients. Settings are hand-placed files; nothing is saved here."""
    return public_settings(load_settings())


@app.get("/api/batches")
def api_batches() -> list[dict[str, Any]]:
    return list_batches()


@app.get("/api/batches/{batch_id}")
def api_batch(batch_id: int) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    return batch


@app.post("/api/batches/{batch_id}/annotator")
def api_batch_annotator(batch_id: int, body: AnnotatorIn) -> dict[str, Any]:
    if get_batch(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    set_batch_annotator(batch_id, body.annotator)
    return get_batch(batch_id)


@app.get("/api/sessions")
def api_sessions() -> list[dict[str, Any]]:
    return list_sessions()


@app.post("/api/sessions/from-path")
def api_session_from_path(body: SessionPathIn) -> dict[str, Any]:
    try:
        return start_session_from_dir(body.annotator, body.name, Path(body.path))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/sessions/upload")
async def api_session_upload(
    annotator: str = Form(...),
    name: str = Form(""),
    files: list[UploadFile] = File(...),
    relpaths: list[str] | None = Form(None),
) -> dict[str, Any]:
    paths = relpaths if isinstance(relpaths, list) else ([relpaths] if relpaths else [])
    payload: list[tuple[str, bytes]] = []
    for index, upload in enumerate(files):
        filename = Path(upload.filename or "").name
        if not filename:
            continue
        rel = paths[index] if index < len(paths) else filename
        payload.append((rel, await upload.read()))
    if not payload:
        raise HTTPException(400, "Choose a folder of tiles.")
    try:
        return start_session_from_upload(annotator, name, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/sessions/gpu")
def api_session_gpu(body: SessionGpuIn) -> dict[str, Any]:
    _require_gpu()
    try:
        return start_gpu_session(body.annotator, body.name, body.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/sessions/gpu-upload")
async def api_session_gpu_upload(
    annotator: str = Form(...),
    name: str = Form(""),
    files: list[UploadFile] = File(...),
    relpaths: list[str] | None = Form(None),
) -> dict[str, Any]:
    _require_gpu()
    paths = relpaths if isinstance(relpaths, list) else ([relpaths] if relpaths else [])
    payload: list[tuple[str, bytes]] = []
    for index, upload in enumerate(files):
        filename = Path(upload.filename or "").name
        if not filename:
            continue
        rel = paths[index] if index < len(paths) else filename
        payload.append((rel, await upload.read()))
    if not payload:
        raise HTTPException(400, "Choose a folder of raw photos.")
    try:
        return start_gpu_session_from_upload(annotator, name, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/prediction-jobs/gpu")
def api_prediction_gpu(body: PredictionPathIn) -> dict[str, Any]:
    _require_gpu()
    try:
        return start_prediction_from_path(body.annotator, body.name, body.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/prediction-jobs/remote")
def api_prediction_remote(body: PredictionPathIn) -> dict[str, Any]:
    _require_gpu()
    try:
        return start_prediction_from_remote_path(body.annotator, body.name, body.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


async def _upload_relpaths(
    files: list[UploadFile],
    relpaths: list[str] | None,
) -> list[tuple[str, bytes]]:
    paths = relpaths if isinstance(relpaths, list) else ([relpaths] if relpaths else [])
    payload: list[tuple[str, bytes]] = []
    for index, upload in enumerate(files):
        raw = upload.filename or ""
        filename = Path(raw).name
        if not filename:
            continue
        rel = paths[index] if index < len(paths) else (raw or filename)
        payload.append((rel, await upload.read()))
    return payload


@app.post("/api/prediction-jobs/gpu-upload")
async def api_prediction_gpu_upload(
    annotator: str = Form(...),
    name: str = Form(""),
    files: list[UploadFile] = File(...),
    relpaths: list[str] | None = Form(None),
) -> dict[str, Any]:
    _require_gpu()
    payload = await _upload_relpaths(files, relpaths)
    if not payload:
        raise HTTPException(400, "Choose photographs or a folder of photographs.")
    try:
        return start_prediction_from_upload(annotator, name, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/prediction-jobs/tiled")
def api_prediction_tiled(body: PredictionPathIn) -> dict[str, Any]:
    _require_gpu()
    try:
        return start_prediction_from_tiled(body.annotator, body.name, body.path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/prediction-jobs/tiled-upload")
async def api_prediction_tiled_upload(
    annotator: str = Form(...),
    name: str = Form(""),
    files: list[UploadFile] = File(...),
    relpaths: list[str] | None = Form(None),
) -> dict[str, Any]:
    _require_gpu()
    payload = await _upload_relpaths(files, relpaths)
    if not payload:
        raise HTTPException(400, "Choose a tiled folder with tiles_foliage.csv.")
    try:
        return start_prediction_from_tiled_upload(annotator, name, payload)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/prediction-jobs/repro")
def api_prediction_repro(body: AnnotatorIn) -> dict[str, Any]:
    _require_gpu()
    try:
        return start_final_test_repro(body.annotator)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/prediction-jobs/{job_id}")
def api_prediction_job(job_id: str) -> dict[str, Any]:
    job = get_prediction_job(job_id)
    if job is None:
        raise HTTPException(404, "Prediction job not found")
    return job


@app.get("/api/prediction-runs")
def api_prediction_runs() -> list[dict[str, Any]]:
    return list_prediction_runs()


@app.get("/api/prediction-runs/{run_id}")
def api_prediction_run(run_id: str) -> dict[str, Any]:
    meta = read_run(run_id)
    if meta is None:
        raise HTTPException(404, "Prediction run not found")
    return meta


@app.get("/api/prediction-runs/{run_id}/plants")
def api_prediction_plants(
    run_id: str,
    high_frac: float = 0.20,
    mid_frac: float = 0.30,
) -> dict[str, Any]:
    meta = read_run(run_id)
    if meta is None:
        raise HTTPException(404, "Prediction run not found")
    return {"run": meta, "plants": export_plants(run_id, high_frac, mid_frac)}


@app.get("/api/prediction-runs/{run_id}/plants/{image}")
def api_prediction_plant(
    run_id: str,
    image: str,
    high_frac: float = 0.20,
    mid_frac: float = 0.30,
) -> dict[str, Any]:
    if read_run(run_id) is None:
        raise HTTPException(404, "Prediction run not found")
    try:
        return plant_detail(run_id, image, high_frac, mid_frac)
    except KeyError as exc:
        raise HTTPException(404, "Plant not found") from exc


@app.get("/api/prediction-runs/{run_id}/export")
def api_prediction_export(
    run_id: str,
    level: str = "plants",
    high_frac: float = 0.20,
    mid_frac: float = 0.30,
) -> StreamingResponse:
    if read_run(run_id) is None:
        raise HTTPException(404, "Prediction run not found")
    if level == "tiles":
        rows = export_tiles(run_id)
        fields = TILE_EXPORT_FIELDS
        filename = f"{run_id}_tile_predictions.csv"
    else:
        rows = export_plants(run_id, high_frac, mid_frac)
        fields = PREDICT_EXPORT_FIELDS
        filename = f"{run_id}_plant_severity.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        io.BytesIO(buffer.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/prediction-runs/{run_id}/compare")
def api_prediction_compare(run_id: str) -> dict[str, Any]:
    if read_run(run_id) is None:
        raise HTTPException(404, "Prediction run not found")
    return compare_run_to_deployment(run_id)


def _prediction_media(run_id: str, kind: str, name: str) -> FileResponse | StreamingResponse:
    if read_run(run_id) is None:
        raise HTTPException(404, "Prediction run not found")
    try:
        source, location = resolve_prediction_media(run_id, kind, name)
        if source == "local":
            path = Path(location)
            if not path.exists():
                raise FileNotFoundError(name)
            return FileResponse(path)
        body = iter_remote_file(str(location))
    except FileNotFoundError as exc:
        raise HTTPException(404, f"{kind} not found") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return StreamingResponse(
        body,
        media_type=media_content_type(name),
        headers={"Cache-Control": "private, max-age=3600"},
    )


@app.get("/media/predictions/{run_id}/original/{image}", response_model=None)
def media_prediction_original(run_id: str, image: str) -> FileResponse | StreamingResponse:
    return _prediction_media(run_id, "original", image)


@app.get("/media/predictions/{run_id}/crop/{image}", response_model=None)
def media_prediction_crop(run_id: str, image: str) -> FileResponse | StreamingResponse:
    return _prediction_media(run_id, "crop", image)


@app.get("/media/predictions/{run_id}/tile/{tile}", response_model=None)
def media_prediction_tile(run_id: str, tile: str) -> FileResponse | StreamingResponse:
    return _prediction_media(run_id, "tile", tile)


@app.post("/api/prepare")
def api_prepare() -> dict[str, Any]:
    raise HTTPException(400, KIT_ONLY)


@app.post("/api/prepare/upload")
def api_prepare_upload() -> dict[str, Any]:
    raise HTTPException(400, KIT_ONLY)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        path = DATA / "jobs" / f"{job_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/batches/{batch_id}/next")
def api_next(batch_id: int, image_id: int | None = None, tile_id: int | None = None) -> dict[str, Any]:
    if get_batch(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    if tile_id is not None:
        tile = get_tile(tile_id)
        if tile is None or tile["batch_id"] != batch_id:
            raise HTTPException(404, "Tile not found")
    else:
        tile = next_unlabeled(batch_id, prefer_image_id=image_id)
        if tile is None:
            return {"done": True, "tile": None, "plant": [], "batch": get_batch(batch_id)}
    return {
        "done": False,
        "tile": tile,
        "plant": plant_tiles(batch_id, int(tile["image_id"])),
        "batch": get_batch(batch_id),
    }


@app.get("/api/batches/{batch_id}/review")
def api_review(
    batch_id: int,
    label: str | None = None,
    image_id: int | None = None,
) -> dict[str, Any]:
    batch = get_batch(batch_id)
    if batch is None:
        raise HTTPException(404, "Batch not found")
    return {
        "batch": batch,
        "tiles": review_tiles(batch_id, label=label, image_id=image_id),
    }


@app.post("/api/label")
def api_label(body: LabelIn) -> dict[str, Any]:
    try:
        tile = add_label(
            body.tile_id,
            body.annotator,
            tissue=body.tissue,
            injury=body.injury,
            curl=body.curl,
            label=body.label,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    nxt = next_unlabeled(int(tile["batch_id"]), prefer_image_id=int(tile["image_id"]))
    plant_id = int(nxt["image_id"]) if nxt else int(tile["image_id"])
    return {
        "tile": tile,
        "next": nxt,
        "plant": plant_tiles(int(tile["batch_id"]), plant_id),
        "batch": get_batch(int(tile["batch_id"])),
        "done": nxt is None,
    }


@app.post("/api/undo")
def api_undo(body: UndoIn) -> dict[str, Any]:
    tile = undo_label(body.batch_id)
    if tile is None:
        raise HTTPException(400, "Nothing to undo")
    return {
        "tile": tile,
        "plant": plant_tiles(body.batch_id, int(tile["image_id"])),
        "batch": get_batch(body.batch_id),
    }


@app.get("/api/batches/{batch_id}/export")
def api_export(batch_id: int) -> StreamingResponse:
    if get_batch(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    _csv_path, rows = write_local_export(batch_id)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    data = buffer.getvalue().encode("utf-8")
    filename = f"batch_{batch_id}_labels.csv"
    return StreamingResponse(
        io.BytesIO(data),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

@app.post("/api/batches/{batch_id}/export-gpu")
def api_export_gpu(batch_id: int) -> dict[str, Any]:
    if get_batch(batch_id) is None:
        raise HTTPException(404, "Batch not found")
    _require_gpu()
    try:
        return export_batch_to_gpu(batch_id)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/media/tile/{tile_id}")
def media_tile(tile_id: int) -> FileResponse:
    tile = get_tile(tile_id)
    if tile is None:
        raise HTTPException(404, "Tile not found")
    path = media_path(tile["tile_relpath"])
    if not path.exists():
        raise HTTPException(404, "Tile file missing")
    return FileResponse(path)


@app.get("/media/crop/{image_id}")
def media_crop(image_id: int) -> FileResponse:
    from annotator.db import session

    with session() as conn:
        row = conn.execute("SELECT crop_relpath FROM images WHERE id = ?", (image_id,)).fetchone()
    if row is None or not row["crop_relpath"]:
        raise HTTPException(404, "Plant crop not found")
    path = media_path(row["crop_relpath"])
    if not path.exists():
        raise HTTPException(404, "Plant crop missing")
    return FileResponse(path)


@app.get("/media/context/{tile_id}")
def media_context(tile_id: int) -> StreamingResponse:
    tile = get_tile(tile_id)
    if tile is None or not tile.get("crop_relpath"):
        raise HTTPException(404, "No plant context for this tile")
    path = media_path(tile["crop_relpath"])
    if not path.exists():
        raise HTTPException(404, "Plant crop missing")
    image = Image.open(path).convert("RGB")
    box_x0 = tile.get("box_x0") or 0
    box_y0 = tile.get("box_y0") or 0
    x = (tile.get("x") or 0) - box_x0
    y = (tile.get("y") or 0) - box_y0
    width = tile.get("width") or 512
    height = tile.get("height") or 512
    orig_w, orig_h = image.size
    x0, y0 = max(0, x), max(0, y)
    x1 = min(orig_w - 1, x + width - 1)
    y1 = min(orig_h - 1, y + height - 1)
    image.thumbnail((720, 900))
    if orig_w > 0 and orig_h > 0 and x1 > x0 and y1 > y0:
        scale_x = image.width / orig_w
        scale_y = image.height / orig_h
        box = (
            int(round(x0 * scale_x)),
            int(round(y0 * scale_y)),
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
        )
        draw = ImageDraw.Draw(image)
        # Draw on the resized image so the outline stays thick on screen.
        # Black halo, then hot yellow, so it reads on green leaves and the white tube.
        for color, stroke in (((0, 0, 0), 16), ((255, 220, 0), 9), ((255, 60, 0), 4)):
            draw.rectangle(box, outline=color, width=stroke)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )
