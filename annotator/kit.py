from __future__ import annotations

import shutil
from pathlib import Path

from annotator.config import DATA, SETTINGS_PATH
from annotator.db import create_batch, init_db, set_batch_status
from annotator.pipeline import ingest_run
from annotator.sessions import set_last_session, write_meta

CACHE = DATA / "cache" / "existing_run"
PHOTOS = DATA / "photos"
KEEP_DIRS = {"cache", "photos"}


def reset_clean_kit() -> tuple[int, int]:
    """Wipe local annotator data down to one unlabeled 627-tile batch."""
    if not (CACHE / "tiles_foliage.csv").exists():
        raise FileNotFoundError(
            f"Missing {CACHE / 'tiles_foliage.csv'}. Keep one copy of the 627 tiles there."
        )
    db_path = DATA / "labels.db"
    if db_path.exists():
        db_path.unlink()
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()
    for child in DATA.iterdir():
        if child.name in KEEP_DIRS or child.name == "photos":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        elif child.name not in {"labels.db", "settings.json"}:
            child.unlink()
    PHOTOS.mkdir(parents=True, exist_ok=True)
    init_db()
    batch_id = create_batch(
        "Foliage tiles",
        "",
        source="import",
        status="preparing",
        session_key="session-1",
    )
    count = ingest_run(batch_id, CACHE)
    if count != 627:
        raise RuntimeError(f"Expected 627 foliage tiles, got {count}")
    set_batch_status(batch_id, "ready")
    write_meta(
        "session-1",
        {
            "batch_id": batch_id,
            "name": "Foliage tiles",
            "annotator": "",
            "tiles": count,
            "source": "kit",
        },
    )
    set_last_session("session-1", batch_id)
    return batch_id, count


def kit_tile_dir() -> Path:
    return CACHE / "foliage_tiles"
