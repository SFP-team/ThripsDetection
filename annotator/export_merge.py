from __future__ import annotations

from typing import Any

EXPORT_FIELDS = [
    "image",
    "tile",
    "rel_y",
    "tissue",
    "injury",
    "curl",
    "label",
    "protocol_version",
    "annotator",
    "labeled_at",
]


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image") or ""), str(row.get("tile") or ""))


def image_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("image") or "") for row in rows if row.get("image")}


def labeled_at(row: dict[str, Any]) -> str:
    # ISO 8601 UTC strings from db.now_iso sort as text. A row with no
    # timestamp counts as oldest so it never beats a dated mark.
    return str(row.get("labeled_at") or "")


def merge_rows(
    local_rows: list[dict[str, Any]],
    remote_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge local marks into the GPU rows. Per (image, tile) the newer labeled_at wins.

    New tiles are appended. A GPU row is replaced only when the local mark is
    strictly newer, so a retry of the same export changes nothing and a mark
    someone else exported later is not clobbered.
    """
    merged = list(remote_rows)
    index = {row_key(row): position for position, row in enumerate(merged)}
    added: list[dict[str, Any]] = []
    updated: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for row in local_rows:
        key = row_key(row)
        position = index.get(key)
        if position is None:
            index[key] = len(merged)
            merged.append(row)
            added.append(row)
        elif labeled_at(row) > labeled_at(merged[position]):
            merged[position] = row
            updated.append(row)
        else:
            kept.append(merged[position])
    return {"rows": merged, "added": added, "updated": updated, "kept": kept}


def decide_export_action(
    local_rows: list[dict[str, Any]],
    remote_sets: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Pick create, append, or a new folder for this export."""
    if not local_rows:
        return {"action": "empty", "remote_path": None, "rows": []}
    if not remote_sets:
        return {"action": "create", "remote_path": "labels.csv", "rows": local_rows}

    local_images = image_set(local_rows)
    for path, remote_rows in remote_sets:
        if image_set(remote_rows) & local_images:
            return {"action": "append", "remote_path": path, **merge_rows(local_rows, remote_rows)}
    return {"action": "new_folder", "remote_path": None, "rows": local_rows}
