from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error

from .common import json_dump
from .evaluate_oof import evaluate as evaluate_tiles
from .train_plant_model import predict


def plant_metrics(rows: pd.DataFrame) -> dict:
    true = rows.human_score.to_numpy(int)
    expected = rows.predicted_expected.to_numpy(float)
    rounded = rows.predicted_score.to_numpy(int)
    has_variation = len(np.unique(true)) > 1
    return {
        "plants": int(len(rows)),
        "score_support": {str(score): int(np.sum(true == score)) for score in range(1, 6)},
        "mae": float(mean_absolute_error(true, expected)),
        "median_absolute_error": float(np.median(np.abs(true - expected))),
        "spearman": float(spearmanr(true, expected).statistic) if has_variation else None,
        "qwk": float(cohen_kappa_score(true, rounded, weights="quadratic")) if has_variation else None,
        "exact_accuracy": float(np.mean(true == rounded)),
        "within_one_accuracy": float(np.mean(np.abs(true - rounded) <= 1)),
        "confusion": confusion_matrix(true, rounded, labels=[1, 2, 3, 4, 5]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--tile-predictions", type=Path, required=True)
    parser.add_argument("--plant-features", type=Path, required=True)
    parser.add_argument("--plant-model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    splits = pd.read_csv(root / "training/artifacts/splits.csv")
    test = splits[splits.split.eq("test")].copy()
    tile_predictions = pd.read_csv(args.tile_predictions)
    labels = pd.read_csv(root / "training_tiles.csv")
    expected_tiles = labels[labels.image.isin(test.image)]
    if tile_predictions.tile.duplicated().any():
        raise ValueError("duplicate final-test tile predictions")
    if set(tile_predictions.tile) != set(expected_tiles.tile):
        raise ValueError("final-test tile prediction coverage does not match the locked split")
    tile_rows = tile_predictions.merge(
        labels[["image", "tile", "collection", "tissue", "injury"]],
        on=["image", "tile"], validate="one_to_one",
    )

    features = pd.read_csv(args.plant_features)
    payload = joblib.load(args.plant_model)
    feature_columns = list(payload["features"])
    missing = set(feature_columns) - set(features.columns)
    if missing:
        raise ValueError(f"plant features are missing {sorted(missing)}")
    probabilities, expected = predict(payload["models"], features[feature_columns])
    plant_predictions = features[["image", "collection"]].copy()
    for index, score in enumerate(range(1, 6)):
        plant_predictions[f"p_score_{score}"] = probabilities[:, index]
    plant_predictions["predicted_expected"] = expected
    plant_predictions["predicted_score"] = np.clip(np.rint(expected), 1, 5).astype(int)
    plant_predictions = plant_predictions.merge(
        test[["image", "human_score"]], on="image", validate="one_to_one"
    ).sort_values("image")

    report = {
        "tile_model": {
            "overall": evaluate_tiles(tile_rows),
            "by_collection": {name: evaluate_tiles(group) for name, group in tile_rows.groupby("collection")},
        },
        "plant_model": {
            "overall": plant_metrics(plant_predictions),
            "by_collection": {name: plant_metrics(group) for name, group in plant_predictions.groupby("collection")},
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    plant_predictions.to_csv(args.output_root / "plant_predictions.csv", index=False)
    json_dump(report, args.output_root / "metrics.json")
    print(report)


if __name__ == "__main__":
    main()
