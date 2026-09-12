from __future__ import annotations

import math
from typing import Any

LOW_CONFIDENCE = 0.60
LOW_FLUSH_TILES = 3
HIGH_FRAC = 0.20
MID_FRAC = 0.30
EVIDENCE_COUNT = 6
EVIDENCE_DISTANCE = 384.0
TILE_SIZE = 512

EXPORT_FIELDS = [
    "rank",
    "image",
    "predicted_expected",
    "predicted_score",
    "severity_group",
    "p_score_1",
    "p_score_2",
    "p_score_3",
    "p_score_4",
    "p_score_5",
    "confidence",
    "needs_review",
    "priority_group",
    "percentile",
    "status",
    "failure",
    "tile_count",
    "effective_flush_area",
    "low_flush_warning",
]

TILE_EXPORT_FIELDS = [
    "image",
    "tile",
    "x",
    "y",
    "rel_y",
    "p_flush",
    "p_healthy",
    "p_mild",
    "p_severe",
    "expected_damage",
]

FAILED_NO_PLANT = "no_plant_detected"
FAILED_NO_FOLIAGE = "no_foliage_tiles"
FAILED_SCORING = "scoring_failed"

LOW_FLUSH_COPY = (
    "The model could not see enough new growth for a reliable severity estimate. "
    "Review this plant manually or capture another photograph."
)


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    number = _as_float(value, None)
    if number is None:
        return default
    return int(round(number))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def severity_group(expected: float) -> str:
    if expected <= 1.49:
        return "Minimal visible damage"
    if expected <= 2.49:
        return "Mild visible damage"
    if expected <= 3.49:
        return "Moderate visible damage"
    if expected <= 4.49:
        return "Severe visible damage"
    return "Very severe visible damage"


def priority_group(rank: int, count: int, high_frac: float = HIGH_FRAC, mid_frac: float = MID_FRAC) -> str:
    if count <= 0 or rank <= 0:
        return ""
    high_cut = max(1, math.ceil(count * high_frac))
    mid_cut = max(high_cut, math.ceil(count * (high_frac + mid_frac)))
    if rank <= high_cut:
        return "high"
    if rank <= mid_cut:
        return "medium"
    return "low"


def damage_percentile(rank: int, count: int) -> float | None:
    if count <= 0 or rank <= 0:
        return None
    return round(100.0 * (count - rank + 1) / count, 1)


def distinct_top_tiles(
    tiles: list[dict[str, Any]],
    count: int = EVIDENCE_COUNT,
    distance: float = EVIDENCE_DISTANCE,
) -> list[dict[str, Any]]:
    ranked = []
    for tile in tiles:
        flush = _as_float(tile.get("p_flush"), 0.0) or 0.0
        expected = _as_float(tile.get("expected_damage"), 0.0) or 0.0
        ranked.append((expected * flush, tile))
    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[dict[str, Any]] = []
    centers: list[tuple[float, float]] = []
    for _score, tile in ranked:
        x = (_as_float(tile.get("x"), 0.0) or 0.0) + TILE_SIZE / 2
        y = (_as_float(tile.get("y"), 0.0) or 0.0) + TILE_SIZE / 2
        if all(math.hypot(x - other[0], y - other[1]) >= distance for other in centers):
            selected.append(tile)
            centers.append((x, y))
        if len(selected) == count:
            break
    return selected


def _index_by_image(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("image") or ""): row for row in rows if row.get("image")}


def _tiles_by_image(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        image = str(row.get("image") or "")
        if not image:
            continue
        grouped.setdefault(image, []).append(row)
    return grouped


def classify_failure(
    image: str,
    source_images: list[str],
    segmented: set[str],
    foliage: set[str],
    scored: set[str],
) -> str | None:
    if image not in source_images:
        return None
    if image in scored and image in foliage:
        return None
    if image not in segmented:
        return FAILED_NO_PLANT
    if image not in foliage:
        return FAILED_NO_FOLIAGE
    return FAILED_SCORING


def empty_failed_row(image: str, failure: str, tile_count: int = 0) -> dict[str, Any]:
    return {
        "rank": None,
        "image": image,
        "predicted_expected": None,
        "predicted_score": None,
        "severity_group": "",
        "p_score_1": None,
        "p_score_2": None,
        "p_score_3": None,
        "p_score_4": None,
        "p_score_5": None,
        "confidence": None,
        "needs_review": True,
        "priority_group": "",
        "percentile": None,
        "status": "failed",
        "failure": failure,
        "tile_count": tile_count,
        "effective_flush_area": None,
        "low_flush_warning": True,
        "warning": LOW_FLUSH_COPY,
    }


def scored_row(
    raw: dict[str, Any],
    features: dict[str, Any] | None,
    tile_count: int,
    rank: int,
    count: int,
    high_frac: float,
    mid_frac: float,
) -> dict[str, Any]:
    expected = _as_float(raw.get("predicted_expected"))
    if expected is None:
        return empty_failed_row(str(raw.get("image") or ""), FAILED_SCORING, tile_count)
    confidence = _as_float(raw.get("confidence"), 0.0) or 0.0
    flush_area = _as_float((features or {}).get("effective_flush_area"))
    low_flush = tile_count < LOW_FLUSH_TILES
    needs_review = _as_bool(raw.get("needs_review")) or confidence < LOW_CONFIDENCE
    return {
        "rank": rank,
        "image": raw.get("image"),
        "predicted_expected": round(expected, 4),
        "predicted_score": _as_int(raw.get("predicted_score")),
        "severity_group": severity_group(expected),
        "p_score_1": _as_float(raw.get("p_score_1")),
        "p_score_2": _as_float(raw.get("p_score_2")),
        "p_score_3": _as_float(raw.get("p_score_3")),
        "p_score_4": _as_float(raw.get("p_score_4")),
        "p_score_5": _as_float(raw.get("p_score_5")),
        "confidence": round(confidence, 4),
        "needs_review": needs_review,
        "priority_group": priority_group(rank, count, high_frac, mid_frac),
        "percentile": damage_percentile(rank, count),
        "status": "scored",
        "failure": "",
        "tile_count": tile_count,
        "effective_flush_area": flush_area,
        "low_flush_warning": low_flush,
        "warning": LOW_FLUSH_COPY if low_flush else "",
    }


def assemble_plants(
    source_images: list[str],
    severity_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]] | None = None,
    tile_rows: list[dict[str, Any]] | None = None,
    image_rows: list[dict[str, Any]] | None = None,
    high_frac: float = HIGH_FRAC,
    mid_frac: float = MID_FRAC,
) -> list[dict[str, Any]]:
    names = [str(name) for name in source_images if name]
    severity = _index_by_image(severity_rows)
    features = _index_by_image(feature_rows or [])
    tiles = _tiles_by_image(tile_rows or [])
    segmented = {str(row.get("image") or "") for row in (image_rows or []) if row.get("image")}
    if not segmented and image_rows is None:
        segmented = set(severity) | set(tiles)
    foliage = {image for image, group in tiles.items() if group}
    if tile_rows is None:
        foliage = set(severity)
    scored_names = []
    for image in names:
        row = severity.get(image)
        if row and image in foliage and _as_float(row.get("predicted_expected")) is not None:
            scored_names.append(image)
    scored_names.sort(
        key=lambda image: (
            -(_as_float(severity[image].get("predicted_expected"), 0.0) or 0.0),
            image,
        )
    )
    scored_set = set(scored_names)
    assembled: list[dict[str, Any]] = []
    for index, image in enumerate(scored_names, start=1):
        assembled.append(
            scored_row(
                severity[image],
                features.get(image),
                len(tiles.get(image) or []),
                index,
                len(scored_names),
                high_frac,
                mid_frac,
            )
        )
    for image in names:
        if image in scored_set:
            continue
        failure = classify_failure(image, names, segmented, foliage, scored_set)
        assembled.append(empty_failed_row(image, failure or FAILED_SCORING, len(tiles.get(image) or [])))
    return assembled


def compare_severity(
    actual_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    atol: float = 1e-4,
) -> dict[str, Any]:
    actual = _index_by_image(actual_rows)
    expected = _index_by_image(expected_rows)
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatches: list[dict[str, Any]] = []
    matched = 0
    for image, want in expected.items():
        got = actual.get(image)
        if got is None:
            continue
        want_expected = _as_float(want.get("predicted_expected"))
        got_expected = _as_float(got.get("predicted_expected"))
        want_score = _as_int(want.get("predicted_score"))
        got_score = _as_int(got.get("predicted_score"))
        expected_ok = (
            want_expected is not None
            and got_expected is not None
            and abs(want_expected - got_expected) <= atol
        )
        score_ok = want_score == got_score
        if expected_ok and score_ok:
            matched += 1
            continue
        mismatches.append(
            {
                "image": image,
                "expected_predicted_expected": want_expected,
                "actual_predicted_expected": got_expected,
                "expected_predicted_score": want_score,
                "actual_predicted_score": got_score,
            }
        )
    return {
        "matched": matched,
        "expected": len(expected),
        "actual": len(actual),
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "ok": not missing and not extra and not mismatches and matched == len(expected),
    }
