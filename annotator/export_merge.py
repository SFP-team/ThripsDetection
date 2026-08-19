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
    "annotator",
    "labeled_at",
]


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("image") or ""), str(row.get("tile") or ""))


def image_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("image") or "") for row in rows if row.get("image")}


def extras_to_append(
    local_rows: list[dict[str, Any]],
    remote_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {row_key(row) for row in remote_rows}
    extras = []
    for row in local_rows:
        if row_key(row) in seen:
            continue
        extras.append(row)
        seen.add(row_key(row))
    return extras


def decide_export_action(
    local_rows: list[dict[str, Any]],
    remote_sets: list[tuple[str, list[dict[str, Any]]]],
) -> dict[str, Any]:
    """Pick create, append, or a new folder. Never overwrite a GPU row."""
    if not local_rows:
        return {"action": "empty", "remote_path": None, "extras": []}
    if not remote_sets:
        return {"action": "create", "remote_path": "labels.csv", "extras": local_rows}

    local_images = image_set(local_rows)
    for path, remote_rows in remote_sets:
        if image_set(remote_rows) & local_images:
            extras = extras_to_append(local_rows, remote_rows)
            return {"action": "append", "remote_path": path, "extras": extras, "remote_rows": remote_rows}
    return {"action": "new_folder", "remote_path": None, "extras": local_rows}


def merged_rows(remote_rows: list[dict[str, Any]], extras: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(remote_rows) + list(extras)
