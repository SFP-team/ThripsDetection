from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .common import json_dump


META = {"image", "fold", "collection", "human_score", "split", "severity_band"}


class ConstantProbability:
    """Pickle-safe smoothed fallback for a threshold absent in one fold."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        p = np.full(len(x), self.probability, dtype=float)
        return np.column_stack((1.0 - p, p))


def fit_thresholds(x: pd.DataFrame, y: np.ndarray) -> list[object]:
    models = []
    for threshold in range(1, 5):
        target = (y > threshold).astype(int)
        if len(np.unique(target)) < 2:
            models.append(ConstantProbability((float(target.sum()) + 1.0) / (len(target) + 2.0)))
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.25, penalty="l2", solver="liblinear", class_weight="balanced", max_iter=2000),
        )
        model.fit(x, target)
        models.append(model)
    return models


def predict(models: list[object], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.stack([model.predict_proba(x)[:, 1] for model in models], axis=1)
    cumulative = np.minimum.accumulate(cumulative, axis=1)
    probabilities = np.column_stack(
        [1 - cumulative[:, 0], cumulative[:, 0] - cumulative[:, 1], cumulative[:, 1] - cumulative[:, 2],
         cumulative[:, 2] - cumulative[:, 3], cumulative[:, 3]]
    ).clip(0, 1)
    expected = probabilities @ np.arange(1, 6)
    return probabilities, expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("training/artifacts/plant_model"))
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    splits = pd.read_csv(args.splits)
    data = features.merge(splits, on=["image", "fold", "collection"], validate="one_to_one")
    data = data[data.split == "dev"].reset_index(drop=True)
    feature_columns = [column for column in features.columns if column not in META]
    x = data[feature_columns]
    y = data.human_score.to_numpy(int)
    oof = np.zeros(len(data), dtype=float)
    for fold in sorted(data.fold.unique()):
        train = data.fold != fold
        valid = data.fold == fold
        models = fit_thresholds(x.loc[train], y[train])
        _, oof[valid] = predict(models, x.loc[valid])
    rounded = np.clip(np.rint(oof), 1, 5).astype(int)
    metrics = {
        "mae": float(mean_absolute_error(y, oof)),
        "median_absolute_error": float(np.median(np.abs(y - oof))),
        "spearman": float(spearmanr(y, oof).statistic),
        "qwk": float(cohen_kappa_score(y, rounded, weights="quadratic")),
        "exact_accuracy": float(np.mean(y == rounded)),
        "within_one_accuracy": float(np.mean(np.abs(y - rounded) <= 1)),
        "confusion": confusion_matrix(y, rounded, labels=[1, 2, 3, 4, 5]).tolist(),
    }
    final_models = fit_thresholds(x, y)
    args.output_root.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": final_models, "features": feature_columns}, args.output_root / "model.joblib")
    pd.DataFrame({"image": data.image, "human_score": y, "oof_expected": oof, "oof_rounded": rounded}).to_csv(
        args.output_root / "oof_predictions.csv", index=False
    )
    json_dump(metrics, args.output_root / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
