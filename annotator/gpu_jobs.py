from __future__ import annotations

import csv
import io
import shutil
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from annotator.config import REPO, Settings, annotator_remote_root, load_settings
from annotator.export_merge import EXPORT_FIELDS, decide_export_action, merged_rows
from annotator.pipeline import (
    IMAGE_SUFFIXES,
    JOBS,
    _job,
    _q,
    _sftp_download_run,
    _ssh_client,
    _ssh_exec,
    _update,
)
from annotator.sessions import (
    finish_tile_session,
    make_key,
    session_dir,
    write_local_export,
)

TILED_MARKER = "tiles_foliage.csv"
TILED_HINT = "That folder is already tiled. Use Local tiles, not GPU photos."


def list_photos(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _require_gpu(settings: Settings) -> None:
    if not settings.ssh_password or not settings.ssh_host or not settings.ssh_user:
        raise RuntimeError("GPU password is missing. Add settings.json or .env on this computer.")


def start_gpu_session(annotator: str, name: str = "", folder: str = "") -> dict[str, Any]:
    settings = load_settings()
    _require_gpu(settings)
    image_dir = Path(folder).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Folder not found: {image_dir}")
    if (image_dir / TILED_MARKER).exists() or any(image_dir.rglob(TILED_MARKER)):
        raise RuntimeError(TILED_HINT)
    photos = list_photos(image_dir)
    if not photos:
        raise FileNotFoundError(f"No JPG/PNG files in {image_dir}")
    label = name.strip() or image_dir.name or "GPU photos"
    key = make_key(label)
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = _job("running", "sending", "Sending photos to the GPU", session_key=key)
    thread = threading.Thread(
        target=_run_gpu_session,
        args=(job_id, annotator, label, key, image_dir, settings),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "session_key": key, "reused": False}


def start_gpu_session_from_upload(
    annotator: str,
    name: str,
    files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    settings = load_settings()
    _require_gpu(settings)
    if any(Path(rel).name == TILED_MARKER for rel, _ in files):
        raise RuntimeError(TILED_HINT)
    label = name.strip() or (Path(files[0][0]).parts[0] if files else "GPU photos")
    key = make_key(label)
    photos = session_dir(key) / "photos"
    if photos.exists():
        shutil.rmtree(photos)
    photos.mkdir(parents=True)
    saved = 0
    for rel, data in files:
        suffix = Path(rel).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        dest = photos / Path(rel).name
        dest.write_bytes(data)
        saved += 1
    if saved == 0:
        raise FileNotFoundError("No JPG/PNG files were uploaded.")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = _job("running", "sending", "Sending photos to the GPU", session_key=key)
    thread = threading.Thread(
        target=_run_gpu_session,
        args=(job_id, annotator, label, key, photos, settings),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "session_key": key, "reused": False}


def _run_gpu_session(
    job_id: str,
    annotator: str,
    label: str,
    key: str,
    image_dir: Path,
    settings: Settings,
) -> None:
    try:
        _update(job_id, status="running", step="sending", detail="Sending photos to the GPU")
        run_dir = _prepare_ssh_unique(job_id, key, image_dir, settings)
        _update(job_id, status="running", step="copying", detail="Opening tiles on this computer")
        result = finish_tile_session(annotator, label, key, run_dir, source="gpu")
        _update(
            job_id,
            status="done",
            step="ready",
            detail=f"{result['tiles']} leaf tiles ready",
            batch_id=result["batch_id"],
            session_key=key,
            tiles=result["tiles"],
        )
    except Exception as exc:  # noqa: BLE001
        _update(job_id, status="error", step="error", detail=str(exc), session_key=key)


def _prepare_ssh_unique(job_id: str, key: str, image_dir: Path, settings: Settings) -> Path:
    ssh = _ssh_client(settings)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    remote_root = annotator_remote_root(settings)
    remote_job = f"{remote_root}/jobs/{stamp}-{job_id}"
    remote_images = f"{remote_job}/images"
    remote_out = f"{remote_job}/run"
    try:
        _update(job_id, status="running", step="sending", detail="Sending photos to the GPU")
        _ssh_exec(ssh, f"mkdir -p {_q(remote_images)} {_q(remote_out)} {_q(remote_root + '/exports')}")
        sftp = ssh.open_sftp()
        for path in list_photos(image_dir):
            sftp.put(str(path), f"{remote_images}/{path.name}")
        for script in ("segment_and_tile.py", "filter_tube_tiles.py"):
            sftp.put(str(REPO / script), f"{remote_job}/{script}")
        sftp.close()

        _update(job_id, status="running", step="cutting", detail="Finding and cutting out each plant")
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
                    _q(remote_out),
                    "--fold",
                    str(settings.fold),
                    "--device",
                    settings.device,
                ]
            ),
        )
        _update(job_id, status="running", step="leaves", detail="Keeping only leaf squares")
        _ssh_exec(
            ssh,
            " ".join(
                [
                    _q(settings.remote_python),
                    _q(f"{remote_job}/filter_tube_tiles.py"),
                    "--images",
                    _q(remote_images),
                    "--run",
                    _q(remote_out),
                ]
            ),
        )
        local_run = session_dir(key) / "incoming"
        _update(job_id, status="running", step="copying", detail="Bringing leaf tiles back")
        _sftp_download_run(ssh, remote_out, local_run)
        return local_run
    finally:
        ssh.close()


def export_batch_to_gpu(batch_id: int) -> dict[str, Any]:
    settings = load_settings()
    _require_gpu(settings)
    local_path, local_rows = write_local_export(batch_id)
    ssh = _ssh_client(settings)
    exports_root = f"{annotator_remote_root(settings)}/exports"
    try:
        _ssh_exec(ssh, f"mkdir -p {_q(exports_root)}")
        remote_sets = _download_remote_csvs(ssh, exports_root)
        decision = decide_export_action(local_rows, remote_sets)
        if decision["action"] == "empty":
            return {
                "action": "empty",
                "detail": "No labels to export yet.",
                "local_path": str(local_path),
                "added": 0,
            }
        if decision["action"] == "create":
            remote = f"{exports_root}/labels.csv"
            _sftp_write_csv(ssh, remote, local_rows)
            return {
                "action": "create",
                "detail": f"Wrote {len(local_rows)} rows to annotator exports.",
                "remote_path": remote,
                "local_path": str(local_path),
                "added": len(local_rows),
            }
        if decision["action"] == "append":
            extras = decision["extras"]
            merged = merged_rows(decision.get("remote_rows") or [], extras)
            _sftp_write_csv(ssh, decision["remote_path"], merged)
            return {
                "action": "append",
                "detail": f"Added {len(extras)} new rows. Existing GPU marks were left as they are.",
                "remote_path": decision["remote_path"],
                "local_path": str(local_path),
                "added": len(extras),
            }
        from annotator.db import get_batch

        batch = get_batch(batch_id) or {}
        folder = batch.get("session_key") or f"session-{batch_id}"
        remote = f"{exports_root}/{folder}/labels.csv"
        _ssh_exec(ssh, f"mkdir -p {_q(exports_root + '/' + folder)}")
        _sftp_write_csv(ssh, remote, local_rows)
        return {
            "action": "new_folder",
            "detail": f"Images did not overlap. Wrote {len(local_rows)} rows to a new GPU folder.",
            "remote_path": remote,
            "local_path": str(local_path),
            "added": len(local_rows),
        }
    finally:
        ssh.close()


def _download_remote_csvs(ssh: Any, exports_root: str) -> list[tuple[str, list[dict[str, Any]]]]:
    sftp = ssh.open_sftp()
    found: list[tuple[str, list[dict[str, Any]]]] = []
    try:
        try:
            names = sftp.listdir(exports_root)
        except FileNotFoundError:
            return []
        candidates = []
        if "labels.csv" in names:
            candidates.append(f"{exports_root}/labels.csv")
        for name in names:
            if name == "labels.csv":
                continue
            remote_dir = f"{exports_root}/{name}"
            try:
                sub = sftp.listdir(remote_dir)
            except OSError:
                continue
            if "labels.csv" in sub:
                candidates.append(f"{remote_dir}/labels.csv")
        for remote in candidates:
            found.append((remote, _sftp_read_csv(sftp, remote)))
    finally:
        sftp.close()
    return found


def _sftp_read_csv(sftp: Any, remote: str) -> list[dict[str, Any]]:
    handle = tempfile.NamedTemporaryFile("w+b", suffix=".csv", delete=False)
    temp = Path(handle.name)
    handle.close()
    try:
        sftp.get(remote, str(temp))
        text = temp.read_text(encoding="utf-8")
    finally:
        temp.unlink(missing_ok=True)
    return list(csv.DictReader(io.StringIO(text)))


def _sftp_write_csv(ssh: Any, remote: str, rows: list[dict[str, Any]]) -> None:
    sftp = ssh.open_sftp()
    handle = tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False)
    temp = Path(handle.name)
    try:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.close()
        sftp.put(str(temp), remote)
    finally:
        handle.close()
        temp.unlink(missing_ok=True)
        sftp.close()
