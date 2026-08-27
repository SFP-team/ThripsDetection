from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def coverage_weights(rows: pd.DataFrame, tile_size: int = 512) -> np.ndarray:
    max_x = int(rows.x.max()) + tile_size
    max_y = int(rows.y.max()) + tile_size
    coverage = np.zeros((max_y, max_x), dtype=np.uint8)
    for row in rows.itertuples():
        coverage[int(row.y):int(row.y) + tile_size, int(row.x):int(row.x) + tile_size] += 1
    inverse = np.zeros_like(coverage, dtype=np.float32)
    np.divide(1.0, coverage, out=inverse, where=coverage > 0)
    result = []
    for row in rows.itertuples():
        result.append(float(inverse[int(row.y):int(row.y) + tile_size, int(row.x):int(row.x) + tile_size].sum()))
    return np.asarray(result)


def distinct_top_mean(rows: pd.DataFrame, values: np.ndarray, count: int = 3, distance: float = 384.0) -> float:
    order = np.argsort(-values)
    selected: list[int] = []
    centers = rows[["x", "y"]].to_numpy(dtype=float) + 256.0
    for index in order:
        if all(np.linalg.norm(centers[index] - centers[other]) >= distance for other in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    return float(values[selected].mean()) if selected else 0.0


def aggregate_one(rows: pd.DataFrame) -> dict:
    coverage = coverage_weights(rows)
    flush_weight = coverage * rows.p_flush.to_numpy(float)
    denom = max(float(flush_weight.sum()), 1e-8)
    expected = rows.expected_damage.to_numpy(float)
    severe = rows.p_severe.to_numpy(float)
    any_damage = 1.0 - rows.p_healthy.to_numpy(float)
    upper = rows.rel_y.to_numpy(float) <= 0.35
    upper_weight = flush_weight * upper
    upper_denom = max(float(upper_weight.sum()), 1e-8)
    normalized = flush_weight / denom
    probability_matrix = rows[["p_healthy", "p_mild", "p_severe"]].to_numpy(dtype=float)
    entropy = -np.sum(
        probability_matrix * np.log(np.clip(probability_matrix, 1e-8, 1.0)), axis=1
    )
    return {
        "image": rows.image.iloc[0],
        "fold": int(rows.fold.iloc[0]),
        "mean_expected_damage": float(np.sum(normalized * expected)),
        "severe_mass": float(np.sum(normalized * severe)),
        "any_damage_mass": float(np.sum(normalized * any_damage)),
        "q75_expected": float(np.quantile(expected, 0.75)),
        "q90_expected": float(np.quantile(expected, 0.90)),
        "top3_distinct_expected": distinct_top_mean(rows.reset_index(drop=True), expected),
        "upper_mean_expected": float(np.sum(upper_weight * expected) / upper_denom),
        "upper_severe_mass": float(np.sum(upper_weight * severe) / upper_denom),
        "effective_flush_area": float(flush_weight.sum() / (512.0 * 512.0)),
        "spatial_spread": float(np.sqrt(np.sum(normalized * (expected - np.sum(normalized * expected)) ** 2))),
        "mean_entropy": float(np.sum(normalized * entropy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--metadata", type=Path,
                        help="Tile manifest; defaults to <dataset-root>/training_tiles.csv")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/oof_plants.csv"))
    args = parser.parse_args()
    metadata = pd.read_csv(args.metadata or (args.dataset_root / "training_tiles.csv"))
    if "decision" in metadata.columns:
        metadata = metadata[metadata.decision.eq("keep")].copy()
    if "collection" not in metadata.columns:
        metadata["collection"] = "field_upload"
    predictions = pd.read_csv(args.predictions)
    rows = predictions.merge(metadata[["image", "tile", "x", "y", "rel_y", "collection"]],
                             on=["image", "tile"], validate="one_to_one")
    result = pd.DataFrame([aggregate_one(group.copy()) for _, group in rows.groupby("image", sort=True)])
    collections = metadata[["image", "collection"]].drop_duplicates()
    result = result.merge(collections, on="image", validate="one_to_one")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} plant feature rows to {args.output}")


if __name__ == "__main__":
    main()
