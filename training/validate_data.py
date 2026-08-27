from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .common import image_stem, load_config, require_columns


VALID_TISSUES = {"flush", "mature", "tube"}
VALID_FLUSH_DAMAGE = {"healthy", "mild", "severe", "uncertain"}


def validate(dataset_root: Path, require_all_scores: bool = False) -> dict:
    cfg = load_config()
    tiles_path = dataset_root / "training_tiles.csv"
    plants_path = dataset_root / "plants.csv"
    scores_path = dataset_root / "human_thrips_scores.csv"

    tiles = pd.read_csv(tiles_path, dtype=str).fillna("")
    plants = pd.read_csv(plants_path, dtype=str).fillna("")
    scores = pd.read_csv(scores_path, dtype=str).fillna("")
    require_columns(
        tiles.columns,
        {"image", "tile", "tile_path", "collection", "rel_y", "x", "y", "tissue", "injury", "protocol_version"},
        str(tiles_path),
    )
    require_columns(plants.columns, {"image", "collection"}, str(plants_path))
    require_columns(scores.columns, {"image", "human_score"}, str(scores_path))

    errors: list[str] = []
    warnings: list[str] = []
    if tiles["tile"].duplicated().any():
        errors.append(f"duplicate tile names: {int(tiles['tile'].duplicated().sum())}")
    if plants["image"].duplicated().any():
        errors.append(f"duplicate plants: {int(plants['image'].duplicated().sum())}")
    if scores["image"].duplicated().any():
        errors.append(f"duplicate score rows: {int(scores['image'].duplicated().sum())}")

    bad_tissue = sorted(set(tiles["tissue"]) - VALID_TISSUES)
    if bad_tissue:
        errors.append(f"invalid tissue labels: {bad_tissue}")
    flush = tiles[tiles["tissue"] == "flush"]
    bad_damage = sorted(set(flush["injury"]) - VALID_FLUSH_DAMAGE)
    if bad_damage:
        errors.append(f"invalid flush damage labels: {bad_damage}")
    nonflush = tiles[tiles["tissue"] != "flush"]
    if not nonflush["injury"].isin(["skip"]).all():
        errors.append("mature/tube rows must have injury=skip")

    protocol = cfg["protocol_version"]
    wrong_protocol = int((tiles["protocol_version"] != protocol).sum())
    if wrong_protocol:
        errors.append(f"{wrong_protocol} tile rows do not use protocol {protocol}")

    tile_images = set(tiles["image"].map(image_stem))
    plant_images = set(plants["image"].map(image_stem))
    if tile_images != plant_images:
        errors.append(
            f"tile/plant image mismatch: tile_only={sorted(tile_images-plant_images)[:5]}, "
            f"plant_only={sorted(plant_images-tile_images)[:5]}"
        )

    missing_files = [
        row.tile_path
        for row in tiles[["tile_path"]].itertuples(index=False)
        if not (dataset_root / row.tile_path).is_file()
    ]
    if missing_files:
        errors.append(f"missing tile files: {len(missing_files)}; first={missing_files[:3]}")

    score_values = pd.to_numeric(scores["human_score"], errors="coerce")
    invalid_scores = scores[score_values.isna() | ~score_values.between(1, 5)]
    if len(invalid_scores):
        errors.append(f"invalid 1-5 score rows: {invalid_scores['image'].tolist()[:10]}")
    scored_images = set(scores["image"].map(image_stem))
    unknown_scored = sorted(scored_images - plant_images)
    if unknown_scored:
        errors.append(f"scores reference unknown plants: {unknown_scored[:10]}")
    missing_scores = sorted(plant_images - scored_images)
    if missing_scores:
        message = f"whole-plant scores still missing for {len(missing_scores)} plants"
        (errors if require_all_scores else warnings).append(message)

    report = {
        "ok": not errors,
        "n_plants": len(plants),
        "n_tiles": len(tiles),
        "n_scores": len(scores),
        "missing_score_count": len(missing_scores),
        "missing_score_images": missing_scores,
        "collections": dict(Counter(plants["collection"])),
        "tissue": dict(Counter(tiles["tissue"])),
        "flush_damage": dict(Counter(flush["injury"])),
        "errors": errors,
        "warnings": warnings,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--require-all-scores", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    report = validate(args.dataset_root.resolve(), args.require_all_scores)
    print(json.dumps(report, indent=2) if args.json else _human_report(report))
    if not report["ok"]:
        raise SystemExit(2)


def _human_report(report: dict) -> str:
    lines = [
        f"Dataset: {report['n_plants']} plants, {report['n_tiles']} tiles, {report['n_scores']} scores",
        f"Collections: {report['collections']}",
        f"Tissue: {report['tissue']}",
        f"Flush damage: {report['flush_damage']}",
    ]
    lines += [f"WARNING: {item}" for item in report["warnings"]]
    lines += [f"ERROR: {item}" for item in report["errors"]]
    lines.append("PASS" if report["ok"] else "FAIL")
    return "\n".join(lines)


if __name__ == "__main__":
    main()

