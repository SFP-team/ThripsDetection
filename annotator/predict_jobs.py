from __future__ import annotations

import csv
import json
import shutil
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

from annotator.config import DATA, REPO, Settings, load_settings
from annotator.gpu_jobs import list_photos
from annotator.pipeline import IMAGE_SUFFIXES, _q, _ssh_client, _ssh_exec
from annotator.predict_report import (
    EXPORT_FIELDS,
    assemble_plants,
    compare_severity,
    distinct_top_tiles,
)
from annotator.sessions import find_tile_run, make_key, safe_relpath

PREDICTIONS = DATA / "predictions"
JOB_STORE = DATA / "prediction_jobs"
PREDICTION_JOBS: dict[str, dict[str, Any]] = {}

TILED_MARKER = "tiles_foliage.csv"
RAW_TILED_HINT = "That folder is already tiled. Use Existing tiled run, not raw photos."
TILED_MISSING = "This folder has no tiled leaf squares (tiles_foliage.csv)."

REMOTE_TRAINING = "/home/fpt/ThripsDetection"
TRAINING_PYTHON = f"{REMOTE_TRAINING}/.training-venv/bin/python"
MODEL_ROOT = f"{REMOTE_TRAINING}/training/artifacts/facebook__dinov3-vits16-pretrain-lvd1689m"
PLANT_MODEL = f"{REMOTE_TRAINING}/training/artifacts/plant_model/model.joblib"
FINAL_TEST_DEPLOYMENT = f"{REMOTE_TRAINING}/training/artifacts/final_test/deployment_predictions.csv"
REMOTE_PREDICTION_STORE = f"{REMOTE_TRAINING}/prediction_runs"

STEP_DETAIL = {
    "waiting": "Waiting for a free GPU",
    "sending": "Sending photographs to the GPU",
    "segmenting": "Finding and cutting out each plant",
    "filtering": "Keeping leaf tiles",
    "scoring_tiles": "Measuring visible damage",
    "aggregating_plants": "Calculating whole-plant severity",
    "ready": "Predictions are ready",
    "error": "Something went wrong",
}

CSV_DOWNLOAD_NAMES = (
    "tiles_foliage.csv",
    "images.csv",
    "tile_predictions.csv",
    "plant_features.csv",
    "plant_severity.csv",
    "predict.log",
    "deployment_predictions.csv",
    "test_tiles.csv",
    "progress.json",
    "segment_progress.json",
)
MEDIA_DOWNLOAD_NAMES = (
    "foliage_tiles",
    "plant_crops",
)
DOWNLOAD_NAMES = CSV_DOWNLOAD_NAMES + MEDIA_DOWNLOAD_NAMES


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prediction_remote_root(settings: Settings) -> str:
    parent = Path(settings.remote_work.rstrip("/")).parent
    return str(parent / "prediction_jobs")


def run_dir(run_id: str) -> Path:
    return PREDICTIONS / run_id


def incoming_dir(run_id: str) -> Path:
    return run_dir(run_id) / "incoming"


def photos_dir(run_id: str) -> Path:
    return run_dir(run_id) / "photos"


def _require_gpu(settings: Settings) -> None:
    if not settings.ssh_password or not settings.ssh_host or not settings.ssh_user:
        raise RuntimeError("GPU password is missing. Add settings.json or .env on this computer.")


def cuda_device_index(device: str) -> str:
    index = device.split(":", 1)[1] if device.startswith("cuda:") else (device or "0")
    return index if index.isdigit() else "0"


def inspect_raw_folder(folder: Path) -> list[Path]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if (folder / TILED_MARKER).exists() or any(folder.rglob(TILED_MARKER)):
        raise RuntimeError(RAW_TILED_HINT)
    photos = list_photos(folder)
    if not photos:
        raise FileNotFoundError(f"No JPG/PNG files in {folder}")
    return unique_basenames(photos)


def inspect_tiled_folder(folder: Path) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    try:
        return find_tile_run(folder)
    except FileNotFoundError as exc:
        raise FileNotFoundError(TILED_MISSING) from exc


def unique_basenames(paths: list[Path]) -> list[Path]:
    seen: dict[str, Path] = {}
    for path in paths:
        name = path.name
        if name in seen:
            raise RuntimeError(f"Duplicate filename in this upload: {name}")
        seen[name] = path
    return list(seen.values())


def save_uploaded_photos(run_id: str, files: list[tuple[str, bytes]]) -> list[Path]:
    dest = photos_dir(run_id)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    saved: list[Path] = []
    names: set[str] = set()
    for rel, data in files:
        suffix = Path(rel).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        name = Path(rel).name
        if name in names:
            raise RuntimeError(f"Duplicate filename in this upload: {name}")
        names.add(name)
        path = dest / name
        path.write_bytes(data)
        saved.append(path)
    if not saved:
        raise FileNotFoundError("No JPG/PNG files were uploaded.")
    return saved


def copy_photos(run_id: str, photos: list[Path]) -> list[Path]:
    dest = photos_dir(run_id)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    copied: list[Path] = []
    for path in unique_basenames(photos):
        target = dest / path.name
        shutil.copy2(path, target)
        copied.append(target)
    return copied


def _job_payload(status: str, step: str, detail: str = "", **extra: Any) -> dict[str, Any]:
    return {"status": status, "step": step, "detail": detail, **extra}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _update_job(job_id: str, run_id: str, **fields: Any) -> None:
    job = PREDICTION_JOBS.setdefault(job_id, {})
    job.update(fields)
    job["job_id"] = job_id
    job["run_id"] = run_id
    _write_json(JOB_STORE / f"{job_id}.json", job)
    _write_json(run_dir(run_id) / "job.json", job)


def get_prediction_job(job_id: str) -> dict[str, Any] | None:
    if job_id in PREDICTION_JOBS:
        return PREDICTION_JOBS[job_id]
    path = JOB_STORE / f"{job_id}.json"
    if path.exists():
        return json.loads(path.read_text())
    if PREDICTIONS.exists():
        for job_path in PREDICTIONS.glob("*/job.json"):
            payload = json.loads(job_path.read_text())
            if payload.get("job_id") == job_id:
                return payload
    return None


def read_run(run_id: str) -> dict[str, Any] | None:
    path = run_dir(run_id) / "run.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def write_run(run_id: str, payload: dict[str, Any]) -> None:
    _write_json(run_dir(run_id) / "run.json", payload)


def list_prediction_runs() -> list[dict[str, Any]]:
    if not PREDICTIONS.exists():
        return []
    runs = []
    for path in PREDICTIONS.iterdir():
        if path.name.startswith("_"):
            continue
        meta = read_run(path.name)
        if meta and meta.get("source") != "fixture":
            runs.append(meta)
    runs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return runs


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def keep_tile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("decision", "keep") == "keep"]


def source_image_names(run_id: str) -> list[str]:
    incoming = incoming_dir(run_id)
    images: list[str] = []
    seen: set[str] = set()

    def add(name: Any) -> None:
        image = str(name or "")
        if image and image not in seen:
            seen.add(image)
            images.append(image)

    for name in ("images.csv", "plant_severity.csv", "tiles_foliage.csv", "tile_predictions.csv"):
        rows = _read_csv(incoming / name)
        if name == "tiles_foliage.csv":
            rows = keep_tile_rows(rows)
        for row in rows:
            add(row.get("image"))
    if images:
        return images
    photos = photos_dir(run_id)
    if photos.is_dir():
        for path in sorted(photos.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                add(path.name)
    return images


def assemble_run_report(run_id: str, high_frac: float = 0.20, mid_frac: float = 0.30) -> list[dict[str, Any]]:
    incoming = incoming_dir(run_id)
    plants = assemble_plants(
        source_image_names(run_id),
        _read_csv(incoming / "plant_severity.csv"),
        _read_csv(incoming / "plant_features.csv"),
        keep_tile_rows(_read_csv(incoming / "tile_predictions.csv"))
        or keep_tile_rows(_read_csv(incoming / "tiles_foliage.csv")),
        _read_csv(incoming / "images.csv"),
        high_frac=high_frac,
        mid_frac=mid_frac,
    )
    _write_csv(run_dir(run_id) / "plant_report.csv", EXPORT_FIELDS, plants)
    meta = read_run(run_id)
    if meta and meta.get("status") in {None, "ready", "error"}:
        meta["images"] = len(plants)
        meta["scored"] = sum(1 for row in plants if row.get("status") == "scored")
        meta["failed"] = sum(1 for row in plants if row.get("status") == "failed")
        write_run(run_id, meta)
    return plants


def plant_tiles(run_id: str, image: str) -> list[dict[str, Any]]:
    incoming = incoming_dir(run_id)
    info = image_info(run_id, image)
    crop_w, crop_h = crop_pixel_size(run_id, image, info)
    predictions = {
        row.get("tile"): row
        for row in _read_csv(incoming / "tile_predictions.csv")
        if row.get("image") == image
    }
    foliage = [
        row
        for row in keep_tile_rows(_read_csv(incoming / "tiles_foliage.csv"))
        if row.get("image") == image
    ]
    if not foliage:
        foliage = [
            row
            for row in keep_tile_rows(_read_csv(incoming / "test_tiles.csv"))
            if row.get("image") == image
        ]
    if not foliage:
        foliage = [
            row
            for row in _read_csv(incoming / "test_tiles.csv")
            if row.get("image") == image
        ]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in foliage:
        tile = str(row.get("tile") or "")
        pred = predictions.get(tile, {})
        item = {
            "image": image,
            "tile": tile,
            "x": row.get("x"),
            "y": row.get("y"),
            "width": row.get("width") or pred.get("width") or 512,
            "height": row.get("height") or pred.get("height") or 512,
            "rel_y": row.get("rel_y"),
            "foliage_mask_frac": row.get("foliage_mask_frac") or pred.get("foliage_mask_frac"),
            "veg_frac": row.get("veg_frac") or pred.get("veg_frac"),
            "p_flush": pred.get("p_flush"),
            "p_healthy": pred.get("p_healthy"),
            "p_mild": pred.get("p_mild"),
            "p_severe": pred.get("p_severe"),
            "expected_damage": pred.get("expected_damage"),
        }
        rect = crop_tile_rect(item, info, crop_w, crop_h)
        if rect:
            item.update(rect)
        merged.append(item)
        seen.add(tile)
    for tile, pred in predictions.items():
        if tile in seen:
            continue
        item = {
            "image": image,
            "tile": tile,
            "x": pred.get("x"),
            "y": pred.get("y"),
            "width": pred.get("width") or 512,
            "height": pred.get("height") or 512,
            "rel_y": pred.get("rel_y"),
            "foliage_mask_frac": pred.get("foliage_mask_frac"),
            "veg_frac": pred.get("veg_frac"),
            "p_flush": pred.get("p_flush"),
            "p_healthy": pred.get("p_healthy"),
            "p_mild": pred.get("p_mild"),
            "p_severe": pred.get("p_severe"),
            "expected_damage": pred.get("expected_damage"),
        }
        rect = crop_tile_rect(item, info, crop_w, crop_h)
        if rect:
            item.update(rect)
        merged.append(item)
    return merged


def image_info(run_id: str, image: str) -> dict[str, Any]:
    for row in _read_csv(incoming_dir(run_id) / "images.csv"):
        if row.get("image") == image:
            return row
    return {"image": image}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def crop_tile_rect(
    tile: dict[str, Any],
    image_row: dict[str, Any],
    crop_w: float,
    crop_h: float,
) -> dict[str, float] | None:
    """Map a tile from original-image pixels into the plant-crop pixel box."""
    box_x0 = _as_float(image_row.get("box_x0"))
    box_y0 = _as_float(image_row.get("box_y0"))
    box_w = _as_float(image_row.get("box_w") or image_row.get("width"), crop_w)
    box_h = _as_float(image_row.get("box_h") or image_row.get("height"), crop_h)
    if crop_w <= 0 or crop_h <= 0 or box_w <= 0 or box_h <= 0:
        return None
    scale_x = crop_w / box_w
    scale_y = crop_h / box_h
    x = (_as_float(tile.get("x")) - box_x0) * scale_x
    y = (_as_float(tile.get("y")) - box_y0) * scale_y
    width = _as_float(tile.get("width"), 512) * scale_x
    height = _as_float(tile.get("height"), 512) * scale_y
    left = max(0.0, x)
    top = max(0.0, y)
    right = min(crop_w, x + width)
    bottom = min(crop_h, y + height)
    if right - left < 1 or bottom - top < 1:
        return None
    return {
        "crop_x": left,
        "crop_y": top,
        "crop_w": right - left,
        "crop_h": bottom - top,
    }


def crop_pixel_size(run_id: str, image: str, info: dict[str, Any]) -> tuple[float, float]:
    path = crop_path(run_id, image)
    if path is not None:
        try:
            from PIL import Image

            with Image.open(path) as im:
                return float(im.width), float(im.height)
        except Exception:
            pass
    width = _as_float(info.get("box_w") or info.get("width"))
    height = _as_float(info.get("box_h") or info.get("height"))
    return (width or 1.0, height or 1.0)


def crop_filename(image: str) -> str:
    return f"{Path(image).stem}_plant.jpg"


def safe_run_file(run_id: str, *parts: str) -> Path:
    root = run_dir(run_id).resolve()
    path = root.joinpath(*parts).resolve()
    path.relative_to(root)
    return path


def start_prediction_from_path(annotator: str, name: str, folder: str) -> dict[str, Any]:
    settings = load_settings()
    photos = inspect_raw_folder(Path(folder))
    _require_gpu(settings)
    return _start_job(annotator, name or Path(folder).name, "photos", settings, photos=photos)


def start_prediction_from_remote_path(annotator: str, name: str, folder: str) -> dict[str, Any]:
    settings = load_settings()
    _require_gpu(settings)
    remote_images, count = inspect_remote_raw_folder(settings, folder)
    return _start_job(
        annotator,
        name or Path(remote_images).name,
        "remote_photos",
        settings,
        remote_images=remote_images,
        image_count=count,
    )


def inspect_remote_raw_folder(settings: Settings, folder: str) -> tuple[str, int]:
    remote = _safe_remote_dir(folder)
    ssh = _ssh_client(settings)
    try:
        out = _ssh_exec(
            ssh,
            (
                f"{_q(settings.remote_python)} - <<'PY'\n"
                "from pathlib import Path\n"
                f"folder = Path({remote!r})\n"
                "suffixes = {'.jpg', '.jpeg', '.png'}\n"
                "if not folder.is_dir():\n"
                "    print('MISSING')\n"
                "elif (folder / 'tiles_foliage.csv').exists():\n"
                "    print('TILED')\n"
                "else:\n"
                "    names = [p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes]\n"
                "    dups = len(names) - len(set(names))\n"
                "    print(f'OK {len(names)} {dups}')\n"
                "PY"
            ),
        ).strip().splitlines()
    finally:
        ssh.close()
    line = (out[-1] if out else "").strip()
    if line == "MISSING":
        raise FileNotFoundError(f"Folder not found on the GPU: {remote}")
    if line == "TILED":
        raise RuntimeError(RAW_TILED_HINT)
    if not line.startswith("OK "):
        raise RuntimeError(f"Could not inspect the GPU folder: {line or 'empty reply'}")
    parts = line.split()
    count = int(parts[1])
    dups = int(parts[2])
    if count == 0:
        raise FileNotFoundError(f"No JPG/PNG files in {remote}")
    if dups:
        raise RuntimeError(f"Duplicate filename in this upload: {remote}")
    return remote, count


def _safe_remote_dir(folder: str) -> str:
    text = (folder or "").strip()
    path = Path(text)
    if not text or not path.is_absolute() or ".." in path.parts:
        raise ValueError("Paste an absolute folder path on the GPU, for example /home/fpt/ThripsDetection/RoverImages.")
    return path.as_posix()


def _safe_leaf_name(name: str) -> str:
    cleaned = Path(name).name
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError("Invalid file name")
    return cleaned


def start_prediction_from_upload(
    annotator: str,
    name: str,
    files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    if any(Path(rel).name == TILED_MARKER for rel, _ in files):
        raise RuntimeError(RAW_TILED_HINT)
    settings = load_settings()
    _require_gpu(settings)
    label = name.strip() or (Path(files[0][0]).parts[0] if files else "Prediction")
    run_id = make_key(label, PREDICTIONS)
    photos = save_uploaded_photos(run_id, files)
    return _start_job(annotator, label, "photos", settings, photos=photos, run_id=run_id)


def start_prediction_from_tiled(annotator: str, name: str, folder: str) -> dict[str, Any]:
    settings = load_settings()
    tiled = inspect_tiled_folder(Path(folder))
    _require_gpu(settings)
    return _start_job(
        annotator,
        name or tiled.name,
        "tiled",
        settings,
        tiled_dir=tiled,
    )


def start_prediction_from_tiled_upload(
    annotator: str,
    name: str,
    files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    if not any(Path(rel).name == TILED_MARKER for rel, _ in files):
        raise FileNotFoundError(TILED_MISSING)
    settings = load_settings()
    _require_gpu(settings)
    label = name.strip() or (Path(files[0][0]).parts[0] if files else "Tiled prediction")
    run_id = make_key(label, PREDICTIONS)
    dest = incoming_dir(run_id)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel, data in files:
        path = dest / safe_relpath(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    tiled = inspect_tiled_folder(dest)
    return _start_job(annotator, label, "tiled", settings, tiled_dir=tiled, run_id=run_id)


def start_final_test_repro(annotator: str = "lab") -> dict[str, Any]:
    settings = load_settings()
    _require_gpu(settings)
    return _start_job(annotator, "final-test-repro", "repro", settings)


def _start_job(
    annotator: str,
    name: str,
    source: str,
    settings: Settings,
    photos: list[Path] | None = None,
    tiled_dir: Path | None = None,
    run_id: str | None = None,
    remote_images: str | None = None,
    image_count: int | None = None,
) -> dict[str, Any]:
    label = name.strip() or "Prediction"
    key = run_id or make_key(label, PREDICTIONS)
    run_dir(key).mkdir(parents=True, exist_ok=True)
    if photos and source == "photos" and not all(path.parent == photos_dir(key) for path in photos):
        photos = copy_photos(key, photos)
    job_id = uuid.uuid4().hex[:12]
    created = now_iso()
    first_step = "waiting" if source == "remote_photos" else "sending"
    write_run(
        key,
        {
            "run_id": key,
            "job_id": job_id,
            "name": label,
            "annotator": annotator,
            "created_at": created,
            "status": "running",
            "source": source,
            "images": image_count if image_count is not None else len(photos or []),
            "remote_images": remote_images,
            "media_source": "remote" if source == "remote_photos" else "local",
        },
    )
    PREDICTION_JOBS[job_id] = _job_payload(
        "running",
        first_step,
        STEP_DETAIL[first_step],
        session_key=key,
        run_id=key,
        source=source,
    )
    _update_job(job_id, key, status="running", step=first_step, detail=STEP_DETAIL[first_step])
    thread = threading.Thread(
        target=_run_prediction_job,
        args=(job_id, key, source, photos or [], tiled_dir, settings, remote_images),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "run_id": key, "source": source}


def _run_prediction_job(
    job_id: str,
    run_id: str,
    source: str,
    photos: list[Path],
    tiled_dir: Path | None,
    settings: Settings,
    remote_images: str | None = None,
) -> None:
    try:
        ssh = _ssh_client(settings)
        try:
            transport = ssh.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            remote_root = (
                REMOTE_PREDICTION_STORE if source == "remote_photos" else prediction_remote_root(settings)
            )
            remote_job = f"{remote_root}/{stamp}-{job_id}"
            remote_run = f"{remote_job}/run"
            meta = read_run(run_id) or {}
            meta["remote_job"] = remote_job
            meta["remote_run"] = remote_run
            if remote_images:
                meta["remote_images"] = remote_images
            write_run(run_id, meta)
            if source == "repro":
                remote_run = _run_repro_remote(ssh, job_id, run_id, remote_job, settings)
            elif source == "tiled":
                if tiled_dir is None:
                    raise RuntimeError("Tiled folder is missing.")
                _upload_tiled_run(ssh, job_id, run_id, tiled_dir, remote_job, remote_run, settings)
                _run_remote_predict(ssh, job_id, run_id, remote_run, settings)
            elif source == "remote_photos":
                if not remote_images:
                    raise RuntimeError("GPU photograph folder is missing.")
                remote_run = _run_remote_photos(ssh, job_id, run_id, remote_images, remote_job, remote_run, settings)
            else:
                _upload_and_segment(ssh, job_id, run_id, photos, remote_job, remote_run, settings)
                _run_remote_predict(ssh, job_id, run_id, remote_run, settings)
            _finish_prediction_pull(ssh, job_id, run_id, remote_run, source)
        finally:
            ssh.close()
        _mark_run_ready(job_id, run_id)
    except Exception as exc:  # noqa: BLE001
        _mark_run_error(job_id, run_id, exc)


def resume_running_prediction_jobs() -> None:
    settings = load_settings()
    if not settings.ssh_password or not settings.ssh_host:
        return
    seen: set[str] = set()
    for job in _iter_stored_jobs():
        if job.get("status") != "running":
            continue
        job_id = str(job.get("job_id") or "")
        run_id = str(job.get("run_id") or "")
        if not job_id or not run_id or job_id in seen:
            continue
        meta = read_run(run_id) or {}
        if not meta.get("remote_job"):
            continue
        seen.add(job_id)
        PREDICTION_JOBS[job_id] = job
        thread = threading.Thread(
            target=_resume_prediction_job,
            args=(job_id, run_id, settings),
            daemon=True,
        )
        thread.start()


def _iter_stored_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    if JOB_STORE.exists():
        for path in JOB_STORE.glob("*.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
    if PREDICTIONS.exists():
        for path in PREDICTIONS.glob("*/job.json"):
            try:
                jobs.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                continue
    return jobs


def _resume_prediction_job(job_id: str, run_id: str, settings: Settings) -> None:
    try:
        if (incoming_dir(run_id) / "plant_severity.csv").exists():
            _mark_run_ready(job_id, run_id)
            return
        ssh = _ssh_client(settings)
        try:
            transport = ssh.get_transport()
            if transport is not None:
                transport.set_keepalive(30)
            meta = read_run(run_id) or {}
            remote_job = str(meta["remote_job"])
            remote_run = str(meta.get("remote_run") or f"{remote_job}/run")
            source = str(meta.get("source") or "remote_photos")
            pid = int(meta.get("remote_pid") or 0) or None
            _poll_remote_pipeline(ssh, job_id, run_id, remote_job, remote_run, pid)
            _finish_prediction_pull(ssh, job_id, run_id, remote_run, source)
        finally:
            ssh.close()
        _mark_run_ready(job_id, run_id)
    except Exception as exc:  # noqa: BLE001
        _mark_run_error(job_id, run_id, exc)


def _finish_prediction_pull(ssh: Any, job_id: str, run_id: str, remote_run: str, source: str) -> None:
    _update_job(
        job_id,
        run_id,
        status="running",
        step="aggregating_plants",
        detail="Bringing scores to this computer. Photographs stay on the GPU."
        if source == "remote_photos"
        else "Bringing results to this computer",
    )
    dest = incoming_dir(run_id)
    if dest.exists():
        shutil.rmtree(dest)
    names = CSV_DOWNLOAD_NAMES if source == "remote_photos" else DOWNLOAD_NAMES
    _sftp_download_prediction(ssh, remote_run, dest, names=names)


def _mark_run_ready(job_id: str, run_id: str) -> None:
    plants = assemble_run_report(run_id)
    failed = [row for row in plants if row.get("status") == "failed"]
    scored = [row for row in plants if row.get("status") == "scored"]
    meta = read_run(run_id) or {}
    meta.update(
        {
            "status": "ready",
            "images": len(plants),
            "scored": len(scored),
            "failed": len(failed),
            "finished_at": now_iso(),
        }
    )
    write_run(run_id, meta)
    _update_job(
        job_id,
        run_id,
        status="ready",
        step="ready",
        detail=STEP_DETAIL["ready"],
        plants=len(plants),
        scored=len(scored),
        failed=len(failed),
    )


def _mark_run_error(job_id: str, run_id: str, exc: Exception) -> None:
    meta = read_run(run_id) or {}
    meta.update({"status": "error", "error": str(exc), "finished_at": now_iso()})
    write_run(run_id, meta)
    _update_job(job_id, run_id, status="error", step="error", detail=str(exc))


def _upload_and_segment(
    ssh: Any,
    job_id: str,
    run_id: str,
    photos: list[Path],
    remote_job: str,
    remote_run: str,
    settings: Settings,
) -> None:
    remote_images = f"{remote_job}/images"
    _update_job(job_id, run_id, status="running", step="sending", detail=STEP_DETAIL["sending"])
    _ssh_exec(ssh, f"mkdir -p {_q(remote_images)} {_q(remote_run)}")
    sftp = ssh.open_sftp()
    try:
        for path in photos:
            sftp.put(str(path), f"{remote_images}/{path.name}")
        for script in ("segment_and_tile.py", "filter_tube_tiles.py"):
            sftp.put(str(REPO / script), f"{remote_job}/{script}")
    finally:
        sftp.close()
    _update_job(job_id, run_id, status="running", step="segmenting", detail=STEP_DETAIL["segmenting"])
    _ssh_exec(
        ssh,
        " ".join(
            [
                _q(settings.remote_python),
                _q(f"{remote_job}/segment_and_tile.py"),
                "--project",
                _q(settings.remote_project),
                "--images",
                _q(remote_images),
                "--output",
                _q(remote_run),
                "--fold",
                str(settings.fold),
                "--device",
                settings.device,
            ]
        ),
    )
    _update_job(job_id, run_id, status="running", step="filtering", detail=STEP_DETAIL["filtering"])
    _ssh_exec(
        ssh,
        " ".join(
            [
                _q(settings.remote_python),
                _q(f"{remote_job}/filter_tube_tiles.py"),
                "--images",
                _q(remote_images),
                "--run",
                _q(remote_run),
            ]
        ),
    )


def _upload_tiled_run(
    ssh: Any,
    job_id: str,
    run_id: str,
    tiled_dir: Path,
    remote_job: str,
    remote_run: str,
    settings: Settings,
) -> None:
    _update_job(job_id, run_id, status="running", step="sending", detail="Sending tiled run to the GPU")
    _ssh_exec(ssh, f"mkdir -p {_q(remote_run)}")
    sftp = ssh.open_sftp()
    try:
        for name in ("tiles_foliage.csv", "images.csv"):
            src = tiled_dir / name
            if src.exists():
                sftp.put(str(src), f"{remote_run}/{name}")
        for folder in ("foliage_tiles", "plant_crops"):
            src = tiled_dir / folder
            if not src.is_dir():
                continue
            _ssh_exec(ssh, f"mkdir -p {_q(remote_run + '/' + folder)}")
            for path in src.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(src).as_posix()
                remote = f"{remote_run}/{folder}/{rel}"
                parent = str(Path(remote).parent)
                _ssh_exec(ssh, f"mkdir -p {_q(parent)}")
                sftp.put(str(path), remote)
    finally:
        sftp.close()
    originals: list[Path] = []
    for folder in (tiled_dir / "photos", tiled_dir / "images"):
        if folder.is_dir():
            originals.extend(
                path
                for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
    originals.extend(
        path
        for path in tiled_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if originals:
        copy_photos(run_id, originals)


def _run_remote_photos(
    ssh: Any,
    job_id: str,
    run_id: str,
    remote_images: str,
    remote_job: str,
    remote_run: str,
    settings: Settings,
) -> str:
    _update_job(job_id, run_id, status="running", step="waiting", detail=STEP_DETAIL["waiting"])
    _ssh_exec(ssh, f"mkdir -p {_q(remote_job)} {_q(remote_run)}")
    sftp = ssh.open_sftp()
    try:
        for script in ("segment_and_tile.py", "filter_tube_tiles.py"):
            sftp.put(str(REPO / script), f"{remote_job}/{script}")
    finally:
        sftp.close()
    predict_script = remote_predict_script(remote_run, settings, python=TRAINING_PYTHON)
    _sftp_write_text(ssh, f"{remote_run}/run_predict.sh", predict_script)
    pipeline = remote_inplace_script(remote_job, remote_run, remote_images, settings)
    _sftp_write_text(ssh, f"{remote_job}/run_pipeline.sh", pipeline)
    pid = _spawn_remote(ssh, f"bash {_q(remote_job + '/run_pipeline.sh')}")
    meta = read_run(run_id) or {}
    meta["remote_pid"] = pid
    write_run(run_id, meta)
    _poll_remote_pipeline(ssh, job_id, run_id, remote_job, remote_run, pid)
    return remote_run


def remote_inplace_script(remote_job: str, remote_run: str, remote_images: str, settings: Settings) -> str:
    gpu = cuda_device_index(settings.device)
    return f"""#!/bin/bash
set -euo pipefail
JOB={_q(remote_job)}
RUN={_q(remote_run)}
IMAGES={_q(remote_images)}
LOG="$JOB/pipeline.log"
SEG={_q(settings.remote_python)}
TRAIN={_q(TRAINING_PYTHON)}
GPU={gpu}
mkdir -p "$RUN"
exec > >(tee -a "$LOG") 2>&1
progress() {{
  "$SEG" - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
Path({remote_job!r} + "/progress.json").write_text(
    json.dumps({{"step": sys.argv[1], "detail": sys.argv[2], "done": False}}) + "\\n"
)
PY
}}
fail() {{
  echo "$1" | tee "$JOB/FAILED"
  progress error "$1"
  exit 1
}}
avail=$(df -Pm "$JOB" | awk 'NR==2{{print $4}}')
if [ "${{avail:-0}}" -lt 8000 ]; then
  fail "The GPU disk is too full to score this folder (${{avail}} MB free)."
fi
progress waiting "Waiting for a free GPU"
while true; do
  free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${{free:-0}}" -gt 8000 ]; then
    echo "GPU$GPU free ${{free}}MiB"
    break
  fi
  echo "WAITING_GPU gpu=$GPU free=${{free}}MiB"
  progress waiting "Another process is using GPU $GPU (${{free}} MiB free). Waiting."
  sleep 30
done
export CUDA_VISIBLE_DEVICES="$GPU"
progress segmenting "Finding and cutting out each plant"
"$SEG" "$JOB/segment_and_tile.py" \\
  --project {_q(settings.remote_project)} \\
  --images "$IMAGES" \\
  --output "$RUN" \\
  --fold {settings.fold} \\
  --device {_q(settings.device)} \\
  --skip-previews --resume || fail "Plant cut failed. See pipeline.log."
progress filtering "Keeping leaf tiles"
"$SEG" "$JOB/filter_tube_tiles.py" \\
  --images "$IMAGES" \\
  --run "$RUN" \\
  --skip-previews || fail "Leaf filter failed. See pipeline.log."
progress scoring_tiles "Measuring visible damage"
bash "$RUN/run_predict.sh" || fail "Scoring failed. See predict.log."
progress ready "Predictions are ready"
echo DONE > "$JOB/DONE"
"""


def _spawn_remote(ssh: Any, command: str) -> int:
    """Start `command` detached on the GPU and return its PID."""
    transport = ssh.get_transport()
    if transport is None:
        raise RuntimeError("Lost the GPU connection before the job started.")
    chan = transport.open_session()
    chan.exec_command(f"nohup {command} >/dev/null 2>&1 & echo $!")
    out = b""
    while True:
        if chan.recv_ready():
            out += chan.recv(4096)
        if chan.exit_status_ready():
            while chan.recv_ready():
                out += chan.recv(4096)
            break
    if chan.recv_exit_status() != 0:
        raise RuntimeError("Could not start the GPU scoring job.")
    text = out.decode("utf-8", errors="replace").strip()
    pid = text.splitlines()[-1].strip() if text else ""
    if not pid.isdigit():
        raise RuntimeError("The GPU scoring job did not start.")
    return int(pid)


STALL_SECONDS = 2 * 60 * 60


def _poll_remote_pipeline(
    ssh: Any,
    job_id: str,
    run_id: str,
    remote_job: str,
    remote_run: str,
    pid: int | None = None,
) -> None:
    while True:
        status = _remote_pipeline_status(ssh, remote_job, remote_run, pid)
        _update_job(
            job_id,
            run_id,
            status="running",
            step=status["step"],
            detail=status["detail"],
        )
        if status["failed"]:
            raise RuntimeError(status["detail"])
        if status["done"]:
            return
        time.sleep(15)


def _remote_pipeline_status(
    ssh: Any,
    remote_job: str,
    remote_run: str,
    pid: int | None = None,
) -> dict[str, Any]:
    """Read progress files on the GPU. Fails the job when its process is gone.

    With a known PID, liveness is authoritative: no DONE, no FAILED, no process
    means the pipeline died (reboot, kill, OOM). Without a PID (jobs started by
    an older build) a stall of STALL_SECONDS with no file change fails instead.
    """
    command = (
        f"{_q(TRAINING_PYTHON)} - <<'PY'\n"
        "import json, os, time\n"
        "from pathlib import Path\n"
        f"job = Path({remote_job!r})\n"
        f"run = Path({remote_run!r})\n"
        f"pid = {int(pid or 0)}\n"
        f"stall = {STALL_SECONDS}\n"
        "payload = {'step': 'waiting', 'detail': 'Starting on the GPU', 'done': False, 'failed': False}\n"
        "failed = job / 'FAILED'\n"
        "if failed.exists():\n"
        "    payload.update(step='error', detail=failed.read_text().strip() or 'GPU job failed', failed=True)\n"
        "    print(json.dumps(payload))\n"
        "    raise SystemExit\n"
        "progress = job / 'progress.json'\n"
        "if progress.exists():\n"
        "    try:\n"
        "        payload.update(json.loads(progress.read_text()))\n"
        "    except Exception:\n"
        "        pass\n"
        "seg = run / 'segment_progress.json'\n"
        "if seg.exists() and payload.get('step') in {None, 'waiting', 'segmenting'}:\n"
        "    try:\n"
        "        info = json.loads(seg.read_text())\n"
        "        payload['step'] = 'segmenting'\n"
        "        payload['detail'] = f\"Cut {info.get('done', 0)} of {info.get('total', '?')} plants\"\n"
        "        if info.get('image'):\n"
        "            payload['detail'] += f\" ({info['image']})\"\n"
        "    except Exception:\n"
        "        pass\n"
        "if (run / 'tiles_foliage.csv').exists() and payload.get('step') == 'segmenting':\n"
        "    payload.update(step='filtering', detail='Keeping leaf tiles')\n"
        "if (run / 'tile_predictions.csv').exists():\n"
        "    payload.update(step='aggregating_plants', detail='Calculating whole-plant severity')\n"
        "if (job / 'DONE').exists() and (run / 'plant_severity.csv').exists():\n"
        "    payload.update(step='aggregating_plants', detail='Scores are ready', done=True)\n"
        "if not payload['done']:\n"
        "    if pid:\n"
        "        try:\n"
        "            os.kill(pid, 0)\n"
        "            alive = True\n"
        "        except ProcessLookupError:\n"
        "            alive = False\n"
        "        except PermissionError:\n"
        "            alive = True\n"
        "        if not alive:\n"
        "            payload.update(step='error', failed=True, detail=\n"
        "                f'The GPU job stopped before it finished. See {job}/pipeline.log.')\n"
        "    else:\n"
        "        files = [job / 'progress.json', job / 'pipeline.log', seg, run / 'predict.log',\n"
        "                 run / 'tiles.csv', run / 'tiles_foliage.csv', run / 'tile_predictions.csv']\n"
        "        stamps = [p.stat().st_mtime for p in files if p.exists()]\n"
        "        newest = max(stamps) if stamps else job.stat().st_mtime\n"
        "        if time.time() - newest > stall:\n"
        "            payload.update(step='error', failed=True, detail=\n"
        "                f'The GPU job has written nothing for {stall // 3600} hours. See {job}/pipeline.log.')\n"
        "print(json.dumps(payload))\n"
        "PY"
    )
    raw = _ssh_exec(ssh, command).strip().splitlines()
    try:
        return json.loads(raw[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read GPU job status: {raw[-3:]}") from exc


def _run_remote_predict(ssh: Any, job_id: str, run_id: str, remote_run: str, settings: Settings) -> None:
    keep_count = _remote_keep_count(ssh, settings, f"{remote_run}/tiles_foliage.csv")
    if keep_count == 0:
        _update_job(
            job_id,
            run_id,
            status="running",
            step="scoring_tiles",
            detail="No leaf tiles were kept. Marking plants as failed.",
        )
        return
    script = remote_predict_script(remote_run, settings)
    remote_script = f"{remote_run}/run_predict.sh"
    _sftp_write_text(ssh, remote_script, script)
    _update_job(job_id, run_id, status="running", step="scoring_tiles", detail=STEP_DETAIL["scoring_tiles"])
    _ssh_exec(ssh, f"bash {_q(remote_script)}")
    _update_job(
        job_id,
        run_id,
        status="running",
        step="aggregating_plants",
        detail=STEP_DETAIL["aggregating_plants"],
    )


def _run_repro_remote(ssh: Any, job_id: str, run_id: str, remote_job: str, settings: Settings) -> str:
    remote_run = f"{remote_job}/run"
    _update_job(job_id, run_id, status="running", step="scoring_tiles", detail="Reproducing the locked final-test scores")
    _ssh_exec(ssh, f"mkdir -p {_q(remote_run)}")
    script = remote_repro_script(remote_run, settings)
    remote_script = f"{remote_job}/run_repro.sh"
    _sftp_write_text(ssh, remote_script, script)
    _ssh_exec(ssh, f"bash {_q(remote_script)}")
    _update_job(
        job_id,
        run_id,
        status="running",
        step="aggregating_plants",
        detail=STEP_DETAIL["aggregating_plants"],
    )
    return remote_run


def remote_predict_script(remote_run: str, settings: Settings, python: str | None = None) -> str:
    python = python or TRAINING_PYTHON
    device = cuda_device_index(settings.device)
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={device}
export HF_HUB_OFFLINE=1
export PYTHONPATH={_q(REMOTE_TRAINING)}
PYTHON={_q(python)}
RUN={_q(remote_run)}
LOG="$RUN/predict.log"
cd {_q(REMOTE_TRAINING)}
"$PYTHON" -m training.predict_ensemble \\
  --dataset-root "$RUN" \\
  --input-csv "$RUN/tiles_foliage.csv" \\
  --checkpoint-root {_q(MODEL_ROOT)} \\
  --stage finetune \\
  --output "$RUN/tile_predictions.csv" >> "$LOG" 2>&1
"$PYTHON" -m training.aggregate_plants \\
  --dataset-root "$RUN" \\
  --metadata "$RUN/tiles_foliage.csv" \\
  --predictions "$RUN/tile_predictions.csv" \\
  --output "$RUN/plant_features.csv" >> "$LOG" 2>&1
"$PYTHON" -m training.predict_plants \\
  --features "$RUN/plant_features.csv" \\
  --model {_q(PLANT_MODEL)} \\
  --output "$RUN/plant_severity.csv" >> "$LOG" 2>&1
"""


def remote_repro_script(remote_run: str, settings: Settings) -> str:
    python = settings.remote_python
    device = cuda_device_index(settings.device)
    return f"""#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES={device}
export HF_HUB_OFFLINE=1
export PYTHONPATH={_q(REMOTE_TRAINING)}
PYTHON={_q(python)}
ROOT={_q(REMOTE_TRAINING)}
RUN={_q(remote_run)}
LOG="$RUN/predict.log"
cd "$ROOT"
"$PYTHON" -m training.predict_ensemble \\
  --dataset-root "$ROOT" \\
  --splits "$ROOT/training/artifacts/splits.csv" \\
  --checkpoint-root {_q(MODEL_ROOT)} \\
  --stage finetune \\
  --output "$RUN/tile_predictions.csv" >> "$LOG" 2>&1
"$PYTHON" - <<'PY' >> "$LOG" 2>&1
from pathlib import Path
import shutil
import pandas as pd
root = Path({REMOTE_TRAINING!r})
run = Path({remote_run!r})
splits = pd.read_csv(root / "training/artifacts/splits.csv")
tiles = pd.read_csv(root / "training_tiles.csv")
test = tiles[tiles.image.isin(splits.loc[splits.split.eq("test"), "image"])]
test.to_csv(run / "test_tiles.csv", index=False)
keep = test.copy()
keep["decision"] = "keep"
keep.to_csv(run / "tiles_foliage.csv", index=False)
tiles_dir = run / "foliage_tiles"
tiles_dir.mkdir(exist_ok=True)
copied = 0
for tile in test.tile.dropna().unique():
    src = root / "foliage_tiles" / Path(str(tile)).name
    if src.exists():
        shutil.copy2(src, tiles_dir / src.name)
        copied += 1
crops = run / "plant_crops"
crops.mkdir(exist_ok=True)
for image in test.image.dropna().unique():
    name = Path(str(image)).stem + "_plant.jpg"
    src = root / "plant_crops" / name
    if src.exists():
        shutil.copy2(src, crops / name)
images_csv = root / "images.csv"
if images_csv.exists():
    images = pd.read_csv(images_csv)
    images[images.image.isin(test.image)].to_csv(run / "images.csv", index=False)
print(f"Wrote {{len(test)}} locked final-test tiles and copied {{copied}} photographs")
PY
"$PYTHON" -m training.aggregate_plants \\
  --dataset-root "$ROOT" \\
  --metadata "$RUN/test_tiles.csv" \\
  --predictions "$RUN/tile_predictions.csv" \\
  --output "$RUN/plant_features.csv" >> "$LOG" 2>&1
"$PYTHON" -m training.predict_plants \\
  --features "$RUN/plant_features.csv" \\
  --model {_q(PLANT_MODEL)} \\
  --output "$RUN/plant_severity.csv" >> "$LOG" 2>&1
cp {_q(FINAL_TEST_DEPLOYMENT)} "$RUN/deployment_predictions.csv"
"$PYTHON" - <<'PY' >> "$LOG" 2>&1
from pathlib import Path
import pandas as pd
import shutil
root = Path({REMOTE_TRAINING!r})
run = Path({remote_run!r})
test = pd.read_csv(run / "test_tiles.csv")
keep = test.copy()
keep["decision"] = "keep"
keep.to_csv(run / "tiles_foliage.csv", index=False)
tiles_dest = run / "foliage_tiles"
tiles_dest.mkdir(exist_ok=True)
copied = 0
for tile in test.tile.dropna().unique():
    src = root / "foliage_tiles" / Path(str(tile)).name
    if src.is_file():
        shutil.copy2(src, tiles_dest / src.name)
        copied += 1
crops_dest = run / "plant_crops"
crops_dest.mkdir(exist_ok=True)
crop_count = 0
for image in test.image.dropna().unique():
    name = Path(str(image)).stem + "_plant.jpg"
    src = root / "plant_crops" / name
    if src.is_file():
        shutil.copy2(src, crops_dest / name)
        crop_count += 1
images_src = root / "images.csv"
if images_src.is_file():
    images = pd.read_csv(images_src)
    images[images.image.isin(test.image)].to_csv(run / "images.csv", index=False)
print(f"Copied {{copied}} tiles and {{crop_count}} plant crops for the UI")
PY
"""


def _remote_keep_count(ssh: Any, settings: Settings, remote_csv: str) -> int:
    command = (
        f"{_q(settings.remote_python)} - <<'PY'\n"
        "import csv\n"
        "from pathlib import Path\n"
        f"path = Path({remote_csv!r})\n"
        "if not path.exists():\n"
        "    print(0)\n"
        "else:\n"
        "    rows = list(csv.DictReader(path.open()))\n"
        "    print(sum(1 for row in rows if row.get('decision', 'keep') == 'keep'))\n"
        "PY"
    )
    out = _ssh_exec(ssh, command).strip().splitlines()
    try:
        return int(out[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Could not count kept tiles: {out}") from exc


def _sftp_write_text(ssh: Any, remote: str, text: str) -> None:
    sftp = ssh.open_sftp()
    try:
        with sftp.file(remote, "w") as handle:
            handle.write(text)
    finally:
        sftp.close()


def _sftp_download_prediction(
    ssh: Any,
    remote_run: str,
    dest: Path,
    names: tuple[str, ...] = DOWNLOAD_NAMES,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    allowed = set(names)
    remote = remote_run.rstrip("/")
    listed = " ".join(names)
    script = (
        "#!/bin/bash\n"
        "set -e\n"
        f"cd {_q(remote)}\n"
        "files=\"\"\n"
        f"for f in {listed}; do [ -e \"$f\" ] && files=\"$files $f\"; done\n"
        "[ -n \"$files\" ] || { echo 'no prediction files to copy' >&2; exit 1; }\n"
        "tar -czf - $files\n"
    )
    _sftp_write_text(ssh, f"{remote}/_pull.sh", script)
    command = f"bash {_q(remote + '/_pull.sh')}"
    stdin, stdout, stderr = ssh.exec_command(command)
    try:
        with tarfile.open(fileobj=stdout, mode="r|gz") as archive:
            for member in archive:
                name = Path(member.name)
                if not name.parts or name.parts[0] not in allowed:
                    continue
                if name.is_absolute() or ".." in name.parts:
                    continue
                archive.extract(member, dest)
    except tarfile.TarError as exc:
        err = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Could not copy prediction files from the server: {err or exc}") from exc
    code = stdout.channel.recv_exit_status()
    if code != 0:
        err = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote prediction copy failed ({code}): {err or command}")


def export_plants(run_id: str, high_frac: float = 0.20, mid_frac: float = 0.30) -> list[dict[str, Any]]:
    return assemble_run_report(run_id, high_frac=high_frac, mid_frac=mid_frac)


def export_tiles(run_id: str) -> list[dict[str, Any]]:
    incoming = incoming_dir(run_id)
    predictions = _read_csv(incoming / "tile_predictions.csv")
    foliage = {
        (row.get("image"), row.get("tile")): row
        for row in keep_tile_rows(_read_csv(incoming / "tiles_foliage.csv"))
    }
    rows = []
    for row in predictions:
        extra = foliage.get((row.get("image"), row.get("tile")), {})
        rows.append(
            {
                "image": row.get("image"),
                "tile": row.get("tile"),
                "x": extra.get("x") or row.get("x"),
                "y": extra.get("y") or row.get("y"),
                "rel_y": extra.get("rel_y") or row.get("rel_y"),
                "p_flush": row.get("p_flush"),
                "p_healthy": row.get("p_healthy"),
                "p_mild": row.get("p_mild"),
                "p_severe": row.get("p_severe"),
                "expected_damage": row.get("expected_damage"),
            }
        )
    if not rows:
        for key, extra in foliage.items():
            rows.append(
                {
                    "image": extra.get("image"),
                    "tile": extra.get("tile"),
                    "x": extra.get("x"),
                    "y": extra.get("y"),
                    "rel_y": extra.get("rel_y"),
                    "p_flush": "",
                    "p_healthy": "",
                    "p_mild": "",
                    "p_severe": "",
                    "expected_damage": "",
                }
            )
    return rows


def plant_detail(run_id: str, image: str, high_frac: float = 0.20, mid_frac: float = 0.30) -> dict[str, Any]:
    plants = {row["image"]: row for row in assemble_run_report(run_id, high_frac, mid_frac)}
    plant = plants.get(image)
    if plant is None:
        raise KeyError(image)
    tiles = plant_tiles(run_id, image)
    evidence = distinct_top_tiles(tiles) if plant.get("status") == "scored" else []
    info = image_info(run_id, image)
    crop_w, crop_h = crop_pixel_size(run_id, image, info)
    info = {**info, "crop_width": crop_w, "crop_height": crop_h}
    encoded_run = quote(run_id, safe="-_.")
    encoded_image = quote(image, safe=".")
    for tile in tiles:
        name = str(tile.get("tile") or "")
        tile["url"] = f"/media/predictions/{encoded_run}/tile/{quote(name, safe='.')}" if name else None
    for tile in evidence:
        name = str(tile.get("tile") or "")
        tile["url"] = f"/media/predictions/{encoded_run}/tile/{quote(name, safe='.')}" if name else None
    remote_media = uses_remote_media(run_id)
    has_original = original_path(run_id, image) is not None or remote_media
    has_crop = crop_path(run_id, image) is not None or remote_media
    has_tile_images = remote_media or any(
        tile_path(run_id, str(tile.get("tile") or "")) is not None for tile in tiles[:12]
    )
    return {
        "run_id": run_id,
        "plant": plant,
        "image": info,
        "tiles": tiles,
        "evidence": evidence,
        "media": {
            "has_original": has_original,
            "has_crop": has_crop,
            "has_tile_images": has_tile_images,
            "original": f"/media/predictions/{encoded_run}/original/{encoded_image}" if has_original else None,
            "crop": f"/media/predictions/{encoded_run}/crop/{encoded_image}" if has_crop else None,
            "source": "remote" if remote_media else "local",
        },
    }


def compare_run_to_deployment(run_id: str) -> dict[str, Any]:
    incoming = incoming_dir(run_id)
    return compare_severity(
        _read_csv(incoming / "plant_severity.csv"),
        _read_csv(incoming / "deployment_predictions.csv"),
    )


def uses_remote_media(run_id: str) -> bool:
    meta = read_run(run_id) or {}
    return meta.get("media_source") == "remote" and bool(meta.get("remote_run") or meta.get("remote_images"))


def remote_media_path(run_id: str, kind: str, name: str) -> str | None:
    meta = read_run(run_id) or {}
    leaf = _safe_leaf_name(name)
    if kind == "original":
        root = str(meta.get("remote_images") or "")
        return f"{root.rstrip('/')}/{leaf}" if root else None
    remote_run = str(meta.get("remote_run") or "").rstrip("/")
    if not remote_run:
        return None
    if kind == "crop":
        return f"{remote_run}/plant_crops/{crop_filename(leaf)}"
    if kind == "tile":
        return f"{remote_run}/foliage_tiles/{leaf}"
    raise ValueError(f"Unknown prediction media kind: {kind}")


def resolve_prediction_media(run_id: str, kind: str, name: str) -> tuple[str, Path | str]:
    if kind == "original":
        local = original_path(run_id, name)
    elif kind == "crop":
        local = crop_path(run_id, name)
    elif kind == "tile":
        local = tile_path(run_id, name)
    else:
        raise ValueError(f"Unknown prediction media kind: {kind}")
    if local is not None:
        return "local", local
    remote = remote_media_path(run_id, kind, name)
    if remote:
        return "remote", remote
    raise FileNotFoundError(name)


_MEDIA_LOCK = threading.Lock()
_MEDIA_SSH: dict[str, Any] = {}


def _media_ssh(settings: Settings, fresh: bool = False) -> Any:
    """One SSH login per GPU host, reused by every media request.

    Each request still opens its own SFTP channel on that transport, so
    parallel thumbnail loads do not serialize on one channel and the
    expensive part (password login) happens once instead of per image.
    """
    key = f"{settings.ssh_user}@{settings.ssh_host}"
    with _MEDIA_LOCK:
        client = _MEDIA_SSH.get(key)
        transport = client.get_transport() if client is not None else None
        if fresh or transport is None or not transport.is_active():
            if client is not None:
                client.close()
            client = _ssh_client(settings)
            live = client.get_transport()
            if live is not None:
                live.set_keepalive(30)
            _MEDIA_SSH[key] = client
        return client


def _media_sftp(settings: Settings) -> Any:
    try:
        return _media_ssh(settings).open_sftp()
    except Exception:  # noqa: BLE001
        return _media_ssh(settings, fresh=True).open_sftp()


def iter_remote_file(remote_path: str) -> Iterator[bytes]:
    """Stream a GPU file. Raises FileNotFoundError before any byte is sent."""
    settings = load_settings()
    _require_gpu(settings)
    sftp = _media_sftp(settings)
    try:
        size = int(sftp.stat(remote_path).st_size or 0)
        handle = sftp.file(remote_path, "rb")
    except FileNotFoundError as exc:
        sftp.close()
        raise FileNotFoundError(remote_path) from exc
    except Exception:
        sftp.close()
        raise
    if size:
        handle.prefetch(size)

    def stream() -> Iterator[bytes]:
        try:
            while True:
                chunk = handle.read(256 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            handle.close()
            sftp.close()

    return stream()


def media_content_type(name: str) -> str:
    suffix = Path(name).suffix.lower()
    if suffix == ".png":
        return "image/png"
    return "image/jpeg"


def original_path(run_id: str, image: str) -> Path | None:
    name = Path(image).name
    candidate = photos_dir(run_id) / name
    if candidate.exists():
        return candidate
    incoming = incoming_dir(run_id)
    for folder in ("photos", "images"):
        alt = incoming / folder / name
        if alt.exists():
            return alt
    return None


def crop_path(run_id: str, image: str) -> Path | None:
    candidate = incoming_dir(run_id) / "plant_crops" / crop_filename(image)
    return candidate if candidate.exists() else None


def tile_path(run_id: str, tile: str) -> Path | None:
    name = Path(tile).name
    incoming = incoming_dir(run_id)
    for folder in ("foliage_tiles", "tiles"):
        candidate = incoming / folder / name
        if candidate.exists():
            return candidate
    return None
