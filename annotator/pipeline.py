from __future__ import annotations

import csv
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from annotator.config import Settings
from annotator.db import DATA, insert_image, insert_tile, session

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


def ingest_run(batch_id: int, run_dir: Path) -> int:
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
    ]
    if not tile_rows:
        raise RuntimeError("No foliage tiles to import.")

    try:
        run_dir.relative_to(DATA)
        inplace = True
    except ValueError:
        inplace = False

    media = DATA / "media" / str(batch_id)
    tiles_out = media / "tiles"
    crops_out = media / "crops"
    if not inplace:
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
                crop_rel = _store_file(src_crop, crops_out / src_crop.name, inplace)
            image_ids[filename] = insert_image(conn, batch_id, filename, info, crop_rel)

        count = 0
        for row in tile_rows:
            src = foliage_dir / row["tile"]
            if not src.exists():
                fallback = run_dir / "tiles" / row["tile"]
                src = fallback if fallback.exists() else src
            if not src.exists():
                continue
            dest_rel = _store_file(src, tiles_out / row["tile"], inplace)
            insert_tile(
                conn,
                batch_id,
                image_ids[row["image"]],
                row,
                dest_rel,
            )
            count += 1
    if count == 0:
        raise RuntimeError("Foliage tile files were listed but not found on disk.")
    return count


def _store_file(src: Path, dest: Path, inplace: bool) -> str:
    if inplace:
        return str(src.resolve().relative_to(DATA))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copy2(src, dest)
    return str(dest.relative_to(DATA))


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


def _ssh_client(settings: Settings) -> Any:
    try:
        import paramiko
    except ImportError as exc:
        raise RuntimeError("Install paramiko to talk to the GPU server: pip install paramiko") from exc
    if not settings.ssh_host:
        raise RuntimeError("GPU server host is not set.")
    if not settings.ssh_password:
        raise RuntimeError(
            "GPU password is missing. Add settings.json or .env on this computer."
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


