from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotator.config import DATA
from annotator.db import (
    create_batch,
    get_batch,
    list_batches,
    set_batch_session_key,
    set_batch_status,
)
from annotator.pipeline import ingest_run

SESSIONS = DATA / "sessions"
LAST_PATH = DATA / "last_session.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def session_dir(key: str) -> Path:
    return SESSIONS / key


def export_dir(key: str) -> Path:
    path = session_dir(key) / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_key(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", (name or "session").strip()).strip("-")
    slug = (slug or "session")[:40]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    key = f"{stamp}_{slug}"
    dest = session_dir(key)
    suffix = 2
    while dest.exists():
        key = f"{stamp}_{slug}-{suffix}"
        dest = session_dir(key)
        suffix += 1
    return key


def safe_relpath(raw: str) -> Path:
    cleaned = raw.replace("\\", "/").lstrip("/")
    parts = [part for part in Path(cleaned).parts if part not in ("", ".", "..")]
    if not parts:
        raise ValueError("Invalid file path")
    return Path(*parts)


def find_tile_run(root: Path) -> Path:
    matches = sorted(root.rglob("tiles_foliage.csv"))
    if not matches:
        raise FileNotFoundError(
            "This folder has no tiled leaf squares (tiles_foliage.csv). "
            "Upload the foliage tiles folder, not only raw rover photos."
        )
    for csv_path in matches:
        parent = csv_path.parent
        if (parent / "foliage_tiles").is_dir() or (parent / "tiles").is_dir():
            return parent
    return matches[0].parent


def write_meta(key: str, payload: dict[str, Any]) -> None:
    dest = session_dir(key)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "exports").mkdir(exist_ok=True)
    (dest / "session.json").write_text(json.dumps(payload, indent=2) + "\n")


def read_meta(key: str) -> dict[str, Any] | None:
    path = session_dir(key) / "session.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def set_last_session(key: str, batch_id: int) -> None:
    LAST_PATH.write_text(json.dumps({"session_key": key, "batch_id": batch_id}) + "\n")


def last_session() -> dict[str, Any] | None:
    if not LAST_PATH.exists():
        return None
    return json.loads(LAST_PATH.read_text())


def ensure_legacy_sessions() -> None:
    SESSIONS.mkdir(parents=True, exist_ok=True)
    for batch in list_batches():
        if batch.get("status") != "ready":
            continue
        key = batch.get("session_key")
        if key and (session_dir(key) / "session.json").exists():
            continue
        key = key or f"session-{batch['id']}"
        write_meta(
            key,
            {
                "batch_id": batch["id"],
                "name": batch["name"],
                "annotator": batch.get("annotator") or "",
                "created_at": batch.get("created_at") or now_iso(),
                "tiles": batch.get("counts", {}).get("tiles"),
                "source": "legacy",
            },
        )
        set_batch_session_key(int(batch["id"]), key)


def start_session_from_upload(
    annotator: str,
    name: str,
    files: list[tuple[str, bytes]],
) -> dict[str, Any]:
    label = name.strip() or (Path(files[0][0]).parts[0] if files else "Session")
    key = make_key(label)
    incoming = save_uploaded_folder(key, files)
    return _finish_session(annotator, label, key, incoming)


def start_session_from_dir(annotator: str, name: str, folder: Path) -> dict[str, Any]:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")
    run_dir = find_tile_run(folder)
    label = name.strip() or folder.name or "Session"
    key = make_key(label)
    incoming = session_dir(key) / "incoming"
    incoming.parent.mkdir(parents=True, exist_ok=True)
    if incoming.exists():
        shutil.rmtree(incoming)
    shutil.copytree(run_dir, incoming)
    return _finish_session(annotator, label, key, incoming)


def _finish_session(annotator: str, label: str, key: str, run_root: Path) -> dict[str, Any]:
    run_dir = find_tile_run(run_root)
    batch_id = create_batch(label, annotator, source="session", status="preparing", session_key=key)
    try:
        count = ingest_run(batch_id, run_dir)
        if count == 0:
            raise RuntimeError("No foliage tile files were found in that folder.")
        set_batch_status(batch_id, "ready")
    except Exception:
        set_batch_status(batch_id, "error", "Could not open tiles from that folder.")
        raise
    write_meta(
        key,
        {
            "batch_id": batch_id,
            "name": label,
            "annotator": annotator,
            "created_at": now_iso(),
            "tiles": count,
            "source": "upload",
        },
    )
    set_last_session(key, batch_id)
    return {"batch_id": batch_id, "session_key": key, "tiles": count, "reused": False}


def save_uploaded_folder(key: str, files: list[tuple[str, bytes]]) -> Path:
    incoming = session_dir(key) / "incoming"
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    written = 0
    for rel, data in files:
        dest = incoming / safe_relpath(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        written += 1
    if written == 0:
        raise FileNotFoundError("No files were uploaded.")
    return incoming


def session_payload(batch: dict[str, Any]) -> dict[str, Any]:
    key = batch.get("session_key") or f"session-{batch['id']}"
    meta = read_meta(key) or {}
    counts = batch.get("counts") or {}
    return {
        "batch_id": batch["id"],
        "session_key": key,
        "name": meta.get("name") or batch["name"],
        "status": batch["status"],
        "created_at": meta.get("created_at") or batch.get("created_at"),
        "counts": counts,
        "folder": str(session_dir(key).relative_to(DATA)) if session_dir(key).exists() else "",
    }


def list_sessions() -> list[dict[str, Any]]:
    ensure_legacy_sessions()
    ready = [item for item in list_batches() if item.get("status") == "ready"]
    return [session_payload(item) for item in ready]


def session_export_folder(batch_id: int) -> Path:
    batch = get_batch(batch_id)
    key = (batch or {}).get("session_key") or f"session-{batch_id}"
    if batch and not batch.get("session_key"):
        set_batch_session_key(batch_id, key)
        write_meta(
            key,
            {
                "batch_id": batch_id,
                "name": batch["name"],
                "annotator": batch.get("annotator") or "",
                "created_at": batch.get("created_at") or now_iso(),
                "source": "legacy",
            },
        )
    return export_dir(key)
