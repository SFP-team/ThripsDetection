from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from annotator.config import DATA

DB_PATH = DATA / "labels.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    annotator TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ready',
    source TEXT NOT NULL DEFAULT 'import',
    error TEXT,
    created_at TEXT NOT NULL,
    session_key TEXT
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    box_x0 INTEGER,
    box_y0 INTEGER,
    box_x1 INTEGER,
    box_y1 INTEGER,
    crop_relpath TEXT,
    UNIQUE(batch_id, filename)
);

CREATE TABLE IF NOT EXISTS tiles (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    image_id INTEGER NOT NULL,
    tile TEXT NOT NULL,
    x INTEGER,
    y INTEGER,
    width INTEGER,
    height INTEGER,
    rel_y REAL,
    foliage_mask_frac REAL,
    veg_frac REAL,
    tile_relpath TEXT NOT NULL,
    UNIQUE(batch_id, tile)
);

CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY,
    tile_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    tissue TEXT,
    injury TEXT,
    curl TEXT,
    annotator TEXT NOT NULL,
    labeled_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS undo_stack (
    id INTEGER PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    label_id INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tiles_batch ON tiles(batch_id, image_id);
CREATE INDEX IF NOT EXISTS idx_labels_tile ON labels(tile_id, id);
CREATE INDEX IF NOT EXISTS idx_undo_batch ON undo_stack(batch_id, id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def connect() -> sqlite3.Connection:
    DATA.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def session() -> Iterator[sqlite3.Connection]:
    conn = connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with session() as conn:
        conn.executescript(SCHEMA)
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(labels)").fetchall()
        }
        for name in ("tissue", "injury", "curl"):
            if name not in columns:
                conn.execute(f"ALTER TABLE labels ADD COLUMN {name} TEXT")
        batch_cols = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(batches)").fetchall()
        }
        if "session_key" not in batch_cols:
            conn.execute("ALTER TABLE batches ADD COLUMN session_key TEXT")


def row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def create_batch(
    name: str,
    annotator: str,
    source: str,
    status: str = "preparing",
    session_key: str | None = None,
) -> int:
    with session() as conn:
        cursor = conn.execute(
            """
            INSERT INTO batches (name, annotator, status, source, created_at, session_key)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (name, annotator, status, source, now_iso(), session_key),
        )
        return int(cursor.lastrowid)


def set_batch_status(batch_id: int, status: str, error: str | None = None) -> None:
    with session() as conn:
        conn.execute(
            "UPDATE batches SET status = ?, error = ? WHERE id = ?",
            (status, error, batch_id),
        )


def set_batch_annotator(batch_id: int, annotator: str) -> None:
    with session() as conn:
        conn.execute("UPDATE batches SET annotator = ? WHERE id = ?", (annotator, batch_id))


def set_batch_session_key(batch_id: int, session_key: str) -> None:
    with session() as conn:
        conn.execute("UPDATE batches SET session_key = ? WHERE id = ?", (session_key, batch_id))


def list_batches() -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute("SELECT * FROM batches ORDER BY id DESC").fetchall()
        batches = []
        for row in rows:
            item = row_dict(row)
            item["counts"] = _counts(conn, int(row["id"]))
            batches.append(item)
        return batches


def find_ready_batch() -> int | None:
    with session() as conn:
        row = conn.execute(
            """
            SELECT id FROM batches
            WHERE status = 'ready'
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        return int(row["id"]) if row else None


def batch_count() -> int:
    with session() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0])


def get_batch(batch_id: int) -> dict[str, Any] | None:
    with session() as conn:
        row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
        if row is None:
            return None
        item = row_dict(row)
        item["counts"] = _counts(conn, batch_id)
        item["images"] = [
            row_dict(image)
            for image in conn.execute(
                "SELECT id, filename FROM images WHERE batch_id = ? ORDER BY filename",
                (batch_id,),
            )
        ]
        return item


def _counts(conn: sqlite3.Connection, batch_id: int) -> dict[str, int]:
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM tiles WHERE batch_id = ?", (batch_id,)
    ).fetchone()["n"]
    labeled = conn.execute(
        """
        SELECT COUNT(DISTINCT tile_id) AS n
        FROM labels
        WHERE tile_id IN (SELECT id FROM tiles WHERE batch_id = ?)
          AND tissue IS NOT NULL
          AND tissue != ''
        """,
        (batch_id,),
    ).fetchone()["n"]
    counts = {
        "healthy": 0,
        "injured": 0,
        "skip": 0,
        "flush": 0,
        "mature": 0,
        "tube": 0,
        "flush_healthy": 0,
        "flush_injured": 0,
        "curl_yes": 0,
        "curl_no": 0,
    }
    for row in conn.execute(
        """
        SELECT latest.injury AS injury, latest.tissue AS tissue, latest.curl AS curl, latest.label AS label
        FROM (
            SELECT tile_id, injury, tissue, curl, label
            FROM labels
            WHERE id IN (
                SELECT MAX(id) FROM labels
                WHERE tile_id IN (SELECT id FROM tiles WHERE batch_id = ?)
                GROUP BY tile_id
            )
        ) AS latest
        """,
        (batch_id,),
    ):
        injury = row["injury"] or row["label"]
        tissue = row["tissue"] or ""
        if injury in counts:
            counts[injury] += 1
        if tissue in counts:
            counts[tissue] += 1
        if tissue == "flush" and injury == "healthy":
            counts["flush_healthy"] += 1
        if tissue == "flush" and injury == "injured":
            counts["flush_injured"] += 1
        if row["curl"] == "yes":
            counts["curl_yes"] += 1
        if row["curl"] == "no":
            counts["curl_no"] += 1
    plants = conn.execute(
        "SELECT COUNT(*) AS n FROM images WHERE batch_id = ?", (batch_id,)
    ).fetchone()["n"]
    return {
        "plants": int(plants),
        "tiles": int(total),
        "labeled": int(labeled),
        "unlabeled": int(total) - int(labeled),
        **counts,
    }


def insert_image(
    conn: sqlite3.Connection,
    batch_id: int,
    filename: str,
    box: dict[str, Any],
    crop_relpath: str | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO images (
            batch_id, filename, box_x0, box_y0, box_x1, box_y1, crop_relpath
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            filename,
            box.get("box_x0"),
            box.get("box_y0"),
            box.get("box_x1"),
            box.get("box_y1"),
            crop_relpath,
        ),
    )
    return int(cursor.lastrowid)


def insert_tile(
    conn: sqlite3.Connection,
    batch_id: int,
    image_id: int,
    row: dict[str, Any],
    tile_relpath: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO tiles (
            batch_id, image_id, tile, x, y, width, height,
            rel_y, foliage_mask_frac, veg_frac, tile_relpath
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            batch_id,
            image_id,
            row["tile"],
            _int(row.get("x")),
            _int(row.get("y")),
            _int(row.get("width")),
            _int(row.get("height")),
            _float(row.get("rel_y")),
            _float(row.get("foliage_mask_frac")),
            _float(row.get("veg_frac")),
            tile_relpath,
        ),
    )
    return int(cursor.lastrowid)


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def current_label(conn: sqlite3.Connection, tile_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM labels WHERE tile_id = ? ORDER BY id DESC LIMIT 1",
        (tile_id,),
    ).fetchone()
    return row_dict(row)


def tile_payload(conn: sqlite3.Connection, tile_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            tiles.*,
            images.filename AS image,
            images.box_x0, images.box_y0, images.box_x1, images.box_y1,
            images.crop_relpath
        FROM tiles
        JOIN images ON images.id = tiles.image_id
        WHERE tiles.id = ?
        """,
        (tile_id,),
    ).fetchone()
    if row is None:
        return None
    payload = row_dict(row)
    payload["current_label"] = current_label(conn, tile_id)
    return payload


def get_tile(tile_id: int) -> dict[str, Any] | None:
    with session() as conn:
        return tile_payload(conn, tile_id)


def plant_index(conn: sqlite3.Connection, batch_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT id, filename FROM images WHERE batch_id = ? ORDER BY filename",
        (batch_id,),
    ).fetchall()
    return [row_dict(row) for row in rows]


def next_unlabeled(batch_id: int, prefer_image_id: int | None = None) -> dict[str, Any] | None:
    with session() as conn:
        extra = ""
        args: list[Any] = [batch_id]
        if prefer_image_id is not None:
            extra = "CASE WHEN tiles.image_id = ? THEN 0 ELSE 1 END,"
            args.append(prefer_image_id)
        row = conn.execute(
            f"""
            SELECT tiles.id
            FROM tiles
            JOIN images ON images.id = tiles.image_id
            WHERE tiles.batch_id = ?
              AND tiles.id NOT IN (
                  SELECT DISTINCT tile_id FROM labels
                  WHERE tissue IS NOT NULL AND tissue != ''
              )
            ORDER BY {extra} images.filename, tiles.y, tiles.x, tiles.id
            LIMIT 1
            """,
            args,
        ).fetchone()
        if row is None:
            return None
        return tile_payload(conn, int(row["id"]))


def plant_tiles(batch_id: int, image_id: int) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute(
            """
            SELECT tiles.id, tiles.tile, tiles.x, tiles.y
            FROM tiles
            WHERE tiles.batch_id = ? AND tiles.image_id = ?
            ORDER BY tiles.y, tiles.x, tiles.id
            """,
            (batch_id, image_id),
        ).fetchall()
        items = []
        for row in rows:
            item = row_dict(row)
            label = current_label(conn, int(row["id"]))
            item["label"] = (label or {}).get("injury") or (label or {}).get("label")
            item["tissue"] = (label or {}).get("tissue")
            item["injury"] = (label or {}).get("injury") or (label or {}).get("label")
            item["curl"] = (label or {}).get("curl")
            items.append(item)
        return items


def review_tiles(batch_id: int, label: str | None = None, image_id: int | None = None) -> list[dict[str, Any]]:
    with session() as conn:
        args: list[Any] = [batch_id]
        where = [
            "tiles.batch_id = ?",
            "labels.id = (SELECT MAX(id) FROM labels WHERE labels.tile_id = tiles.id)",
            "labels.tissue IS NOT NULL",
            "labels.tissue != ''",
        ]
        if label:
            where.append("COALESCE(labels.injury, labels.label) = ?")
            args.append(label)
        if image_id is not None:
            where.append("tiles.image_id = ?")
            args.append(image_id)
        rows = conn.execute(
            f"""
            SELECT
                tiles.id, tiles.tile, tiles.image_id,
                images.filename AS image,
                COALESCE(labels.injury, labels.label) AS label,
                labels.tissue, labels.injury, labels.curl,
                labels.annotator, labels.labeled_at
            FROM tiles
            JOIN images ON images.id = tiles.image_id
            JOIN labels ON labels.tile_id = tiles.id
            WHERE {' AND '.join(where)}
            ORDER BY images.filename, tiles.y, tiles.x
            """,
            args,
        ).fetchall()
        return [row_dict(row) for row in rows]


TISSUES = {"flush", "mature", "tube"}
INJURIES = {"healthy", "injured", "skip"}
CURLS = {"yes", "no"}


def normalize_annotation(tissue: str, injury: str | None, curl: str | None) -> tuple[str, str, str | None]:
    if tissue not in TISSUES:
        raise ValueError("tissue must be flush, mature, or tube")
    if tissue != "flush":
        return tissue, "skip", None
    if injury not in INJURIES:
        raise ValueError("flush tiles need healthy, injured, or skip")
    if injury == "skip":
        return tissue, injury, None
    if curl not in CURLS:
        raise ValueError("flush healthy/injured tiles need curl yes or no")
    return tissue, injury, curl


def add_label(
    tile_id: int,
    annotator: str,
    tissue: str,
    injury: str | None = None,
    curl: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    tissue, injury, curl = normalize_annotation(tissue, injury or label, curl)
    with session() as conn:
        tile = conn.execute("SELECT batch_id FROM tiles WHERE id = ?", (tile_id,)).fetchone()
        if tile is None:
            raise KeyError(f"tile {tile_id} not found")
        cursor = conn.execute(
            """
            INSERT INTO labels (tile_id, label, tissue, injury, curl, annotator, labeled_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (tile_id, injury, tissue, injury, curl, annotator, now_iso()),
        )
        label_id = int(cursor.lastrowid)
        conn.execute(
            "INSERT INTO undo_stack (batch_id, label_id) VALUES (?, ?)",
            (int(tile["batch_id"]), label_id),
        )
        return tile_payload(conn, tile_id)


def undo_label(batch_id: int) -> dict[str, Any] | None:
    with session() as conn:
        row = conn.execute(
            "SELECT * FROM undo_stack WHERE batch_id = ? ORDER BY id DESC LIMIT 1",
            (batch_id,),
        ).fetchone()
        if row is None:
            return None
        label_row = conn.execute(
            "SELECT tile_id FROM labels WHERE id = ?", (int(row["label_id"]),)
        ).fetchone()
        conn.execute("DELETE FROM labels WHERE id = ?", (int(row["label_id"]),))
        conn.execute("DELETE FROM undo_stack WHERE id = ?", (int(row["id"]),))
        if label_row is None:
            return None
        return tile_payload(conn, int(label_row["tile_id"]))


def export_rows(batch_id: int) -> list[dict[str, Any]]:
    with session() as conn:
        rows = conn.execute(
            """
            SELECT
                images.filename AS image,
                tiles.tile,
                tiles.rel_y,
                COALESCE(labels.tissue, '') AS tissue,
                COALESCE(labels.injury, labels.label) AS injury,
                COALESCE(labels.curl, '') AS curl,
                COALESCE(labels.injury, labels.label) AS label,
                labels.annotator,
                labels.labeled_at
            FROM tiles
            JOIN images ON images.id = tiles.image_id
            JOIN labels ON labels.tile_id = tiles.id
            WHERE tiles.batch_id = ?
              AND labels.id = (
                  SELECT MAX(id) FROM labels WHERE labels.tile_id = tiles.id
              )
              AND labels.tissue IS NOT NULL
              AND labels.tissue != ''
            ORDER BY images.filename, tiles.tile
            """,
            (batch_id,),
        ).fetchall()
        return [row_dict(row) for row in rows]


def media_path(relpath: str) -> Path:
    return DATA / relpath
