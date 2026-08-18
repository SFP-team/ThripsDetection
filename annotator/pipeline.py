from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tarfile
import threading
import uuid
from pathlib import Path
from typing import Any

from annotator.config import REPO, Settings, load_settings
from annotator.db import (
    DATA,
    create_batch,
    insert_image,
    insert_tile,
    session,
    set_batch_status,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
JOBS: dict[str, dict[str, Any]] = {}


def _job(status: str, step: str, detail: str = "", **extra: Any) -> dict[str, Any]:
    payload = {"status": status, "step": step, "detail": detail, **extra}
    return payload


def get_job(job_id: str) -> dict[str, Any] | None:
    return JOBS.get(job_id)


def _update(job_id: str, **fields: Any) -> None:
    job = JOBS.setdefault(job_id, {})
    job.update(fields)
    job_path = DATA / "jobs" / f"{job_id}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job, indent=2) + "\n")


def start_import(
    annotator: str,
    name: str = "",
    local_path: str = "",
    use_remote: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    batch_name = name.strip() or "Existing foliage tiles"
    batch_id = create_batch(batch_name, annotator, source="import", status="preparing")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = _job("running", "copying", "Starting import", batch_id=batch_id)
    thread = threading.Thread(
        target=_run_import,
        args=(job_id, batch_id, annotator, local_path, use_remote, settings),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "batch_id": batch_id}


def start_prepare(
    annotator: str,
    name: str = "",
    folder: str = "",
    uploaded: Path | None = None,
) -> dict[str, Any]:
    settings = load_settings()
    batch_name = name.strip() or Path(folder or uploaded or "new-photos").name
    batch_id = create_batch(batch_name, annotator, source="prepare", status="preparing")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = _job("running", "finding", "Starting prepare", batch_id=batch_id)
    thread = threading.Thread(
        target=_run_prepare,
        args=(job_id, batch_id, annotator, folder, uploaded, settings),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "batch_id": batch_id}


def _run_import(
    job_id: str,
    batch_id: int,
    annotator: str,
    local_path: str,
    use_remote: bool,
    settings: Settings,
) -> None:
    try:
        run_dir = _resolve_run_dir(job_id, batch_id, local_path, use_remote, settings)
        _update(job_id, status="running", step="copying", detail="Reading foliage tiles")
        count = ingest_run(batch_id, run_dir)
        set_batch_status(batch_id, "ready")
        _update(
            job_id,
            status="done",
            step="ready",
            detail=f"{count} leaf tiles ready",
            batch_id=batch_id,
            tiles=count,
        )
    except Exception as exc:  # noqa: BLE001
        set_batch_status(batch_id, "error", str(exc))
        _update(job_id, status="error", step="error", detail=str(exc), batch_id=batch_id)


def _run_prepare(
    job_id: str,
    batch_id: int,
    annotator: str,
    folder: str,
    uploaded: Path | None,
    settings: Settings,
) -> None:
    try:
        image_dir = Path(folder).expanduser().resolve() if folder else uploaded
        if image_dir is None or not image_dir.is_dir():
            raise FileNotFoundError("Choose a folder of rover photos.")
        photos = _list_images(image_dir)
        if not photos:
            raise FileNotFoundError(f"No JPG/PNG files in {image_dir}")

        existing = _reuse_existing(photos, settings)
        if existing is not None:
            _update(job_id, status="running", step="copying", detail="Reusing tiles already made")
            count = ingest_run(batch_id, existing, keep_names={path.name for path in photos})
        elif settings.pipeline_mode == "local":
            run_dir = _prepare_local(job_id, batch_id, image_dir, settings)
            count = ingest_run(batch_id, run_dir)
        else:
            run_dir = _prepare_ssh(job_id, batch_id, image_dir, settings)
            count = ingest_run(batch_id, run_dir)

        set_batch_status(batch_id, "ready")
        _update(
            job_id,
            status="done",
            step="ready",
            detail=f"{count} leaf tiles ready",
            batch_id=batch_id,
            tiles=count,
        )
    except Exception as exc:  # noqa: BLE001
        set_batch_status(batch_id, "error", str(exc))
        _update(job_id, status="error", step="error", detail=str(exc), batch_id=batch_id)


def _list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _resolve_run_dir(
    job_id: str,
    batch_id: int,
    local_path: str,
    use_remote: bool,
    settings: Settings,
) -> Path:
    if local_path:
        path = Path(local_path).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Folder not found: {path}")
        return path
    if settings.local_existing_run:
        path = Path(settings.local_existing_run).expanduser().resolve()
        if path.is_dir() and (path / "tiles_foliage.csv").exists():
            return path
    if use_remote or settings.pipeline_mode == "ssh":
        return _pull_remote_run(job_id, batch_id, settings)
    raise FileNotFoundError(
        "No existing tile folder found. Set a local path or enable the GPU server."
    )


def _reuse_existing(photos: list[Path], settings: Settings) -> Path | None:
    candidates = []
    if settings.local_existing_run:
        candidates.append(Path(settings.local_existing_run).expanduser())
    local_default = DATA / "cache" / "existing_run"
    if local_default.is_dir():
        candidates.append(local_default)
    names = {path.name for path in photos}
    for candidate in candidates:
        csv_path = candidate / "tiles_foliage.csv"
        if not csv_path.exists():
            continue
        rows = list(csv.DictReader(csv_path.open()))
        kept = {row["image"] for row in rows if row.get("decision") == "keep"}
        if names <= kept or names <= {row["image"] for row in rows}:
            return candidate
    return None


def ingest_run(batch_id: int, run_dir: Path, keep_names: set[str] | None = None) -> int:
    run_dir = run_dir.resolve()
    foliage_csv = run_dir / "tiles_foliage.csv"
    images_csv = run_dir / "images.csv"
    if not foliage_csv.exists():
        raise FileNotFoundError(f"Missing {foliage_csv}")
    if not images_csv.exists():
        raise FileNotFoundError(f"Missing {images_csv}")

    image_info = {row["image"]: row for row in csv.DictReader(images_csv.open())}
    tile_rows = [
        row
        for row in csv.DictReader(foliage_csv.open())
        if row.get("decision", "keep") == "keep"
        and (keep_names is None or row["image"] in keep_names)
    ]
    if not tile_rows:
        raise RuntimeError("No foliage tiles to import.")

    media = DATA / "media" / str(batch_id)
    tiles_out = media / "tiles"
    crops_out = media / "crops"
    tiles_out.mkdir(parents=True, exist_ok=True)
    crops_out.mkdir(parents=True, exist_ok=True)

    foliage_dir = run_dir / "foliage_tiles"
    crop_dir = run_dir / "plant_crops"

    with session() as conn:
        image_ids: dict[str, int] = {}
        for filename in sorted({row["image"] for row in tile_rows}):
            info = image_info.get(filename, {})
            stem = Path(filename).stem
            src_crop = crop_dir / f"{stem}_plant.jpg"
            crop_rel = None
            if src_crop.exists():
                dest = crops_out / src_crop.name
                if not dest.exists():
                    shutil.copy2(src_crop, dest)
                crop_rel = str(dest.relative_to(DATA))
            image_ids[filename] = insert_image(conn, batch_id, filename, info, crop_rel)

        count = 0
        for row in tile_rows:
            src = foliage_dir / row["tile"]
            if not src.exists():
                fallback = run_dir / "tiles" / row["tile"]
                src = fallback if fallback.exists() else src
            if not src.exists():
                continue
            dest = tiles_out / row["tile"]
            if not dest.exists():
                shutil.copy2(src, dest)
            insert_tile(
                conn,
                batch_id,
                image_ids[row["image"]],
                row,
                str(dest.relative_to(DATA)),
            )
            count += 1
    if count == 0:
        raise RuntimeError("Foliage tile files were listed but not found on disk.")
    return count


def _prepare_local(job_id: str, batch_id: int, image_dir: Path, settings: Settings) -> Path:
    project = Path(settings.local_project or "").expanduser()
    if not project.is_dir():
        raise FileNotFoundError("Local segmentation project path is not set.")
    out = DATA / "runs" / str(batch_id)
    out.mkdir(parents=True, exist_ok=True)
    python = settings.local_python
    _update(job_id, status="running", step="finding", detail="Finding each plant")
    _run(
        [
            python,
            str(REPO / "segment_and_tile.py"),
            "--project",
            str(project),
            "--images",
            str(image_dir),
            "--output",
            str(out),
            "--fold",
            str(settings.fold),
            "--device",
            settings.device,
        ]
    )
    _update(job_id, status="running", step="cutting", detail="Cutting plants from the background")
    _update(job_id, status="running", step="leaves", detail="Keeping only leaf squares")
    _run(
        [
            python,
            str(REPO / "filter_tube_tiles.py"),
            "--images",
            str(image_dir),
            "--run",
            str(out),
        ]
    )
    return out


def _prepare_ssh(job_id: str, batch_id: int, image_dir: Path, settings: Settings) -> Path:
    ssh = _ssh_client(settings)
    remote_job = f"{settings.remote_work.rstrip('/')}/batch_{batch_id}"
    remote_images = f"{remote_job}/images"
    remote_out = f"{remote_job}/run"
    try:
        _update(job_id, status="running", step="finding", detail="Sending photos to the GPU server")
        _ssh_exec(ssh, f"mkdir -p { _q(remote_images) } { _q(remote_out) }")
        sftp = ssh.open_sftp()
        for path in _list_images(image_dir):
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
        local_run = DATA / "runs" / str(batch_id)
        _update(job_id, status="running", step="copying", detail="Bringing leaf tiles back")
        _sftp_download_run(ssh, remote_out, local_run)
        return local_run
    finally:
        ssh.close()


def _pull_remote_run(job_id: str, batch_id: int, settings: Settings) -> Path:
    if not settings.remote_existing_run:
        raise FileNotFoundError("No remote existing-run path is set.")
    _update(job_id, status="running", step="copying", detail="Copying tiles from the GPU server")
    ssh = _ssh_client(settings)
    dest = DATA / "cache" / "existing_run"
    try:
        _sftp_download_run(ssh, settings.remote_existing_run, dest)
    finally:
        ssh.close()
    return dest


def _sftp_download_run(ssh: Any, remote_run: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    allowed = {"tiles_foliage.csv", "images.csv", "foliage_tiles", "plant_crops"}
    remote = remote_run.rstrip("/")
    command = (
        f"tar -C {_q(remote)} -czf - "
        "tiles_foliage.csv images.csv foliage_tiles plant_crops"
    )
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
        raise RuntimeError(f"Could not copy tiles from the server: {err or exc}") from exc
    code = stdout.channel.recv_exit_status()
    if code != 0:
        err = stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Remote copy failed ({code}): {err or command}")


def _sftp_download_dir(sftp: Any, remote_dir: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        entries = sftp.listdir(remote_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Remote folder missing: {remote_dir}") from exc
    for name in entries:
        if name.startswith("."):
            continue
        _sftp_get(sftp, f"{remote_dir}/{name}", dest / name)


def _sftp_get(sftp: Any, remote: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(dest))


def _ssh_client(settings: Settings) -> Any:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("Install paramiko to talk to the GPU server: pip install paramiko") from exc
    if not settings.ssh_host:
        raise RuntimeError("GPU server host is not set.")
    if not settings.ssh_password:
        raise RuntimeError(
            "GPU password is missing. Open Server settings, save the password, then try again."
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(
            hostname=settings.ssh_host,
            username=settings.ssh_user,
            password=settings.ssh_password,
            timeout=30,
            look_for_keys=False,
            allow_agent=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not sign in to the GPU server. Check host, user, and password in Server settings."
        ) from exc
    return client


def _ssh_exec(ssh: Any, command: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(command, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"Remote command failed ({code}): {command}\n{err or out}")
    return out


def _q(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Command failed")
