from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from .common import json_dump


DAMAGE_ID = {"healthy": 0, "mild": 1, "severe": 2}


def evaluate(rows: pd.DataFrame) -> dict:
    tissue_true = rows.tissue.eq("flush").astype(int).to_numpy()
    tissue_pred = rows.p_flush.ge(0.5).astype(int).to_numpy()
    damage = rows[rows.injury.isin(DAMAGE_ID)].copy()
    true = damage.injury.map(DAMAGE_ID).to_numpy()
    prob = damage[["p_healthy", "p_mild", "p_severe"]].to_numpy(float)
    pred = prob.argmax(axis=1)
    present = sorted(np.unique(true).tolist())
    confusion = confusion_matrix(true, pred, labels=[0, 1, 2])
    recalls = {}
    for index, name in enumerate(["healthy", "mild", "severe"]):
        denominator = confusion[index].sum()
        recalls[name] = float(confusion[index, index] / denominator) if denominator else None
    return {
        "tiles": int(len(rows)),
        "damage_tiles": int(len(damage)),
        "damage_support": {name: int((true == index).sum()) for index, name in enumerate(["healthy", "mild", "severe"])},
        "damage_macro_f1_all_classes": float(f1_score(true, pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "damage_macro_f1_present_classes": float(f1_score(true, pred, average="macro", labels=present, zero_division=0)),
        "damage_balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "damage_qwk": float(cohen_kappa_score(true, pred, weights="quadratic")) if len(present) > 1 else None,
        "recall": recalls,
        "healthy_to_severe_rate": float(np.mean(pred[true == 0] == 2)) if np.any(true == 0) else None,
        "severe_to_healthy_rate": float(np.mean(pred[true == 2] == 0)) if np.any(true == 2) else None,
        "confusion": confusion.tolist(),
        "tissue_f1": float(f1_score(tissue_true, tissue_pred, zero_division=0)),
        "tissue_balanced_accuracy": float(balanced_accuracy_score(tissue_true, tissue_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    predictions = pd.read_csv(args.predictions)
    labels = pd.read_csv(root / "training_tiles.csv")
    splits = pd.read_csv(root / "training/artifacts/splits.csv")
    expected = labels.merge(splits[["image", "split", "fold"]], on="image")
    expected = expected[expected.split.eq("dev")]
    if predictions.tile.duplicated().any():
        raise ValueError("duplicate OOF tile predictions")
    if set(predictions.tile) != set(expected.tile):
        raise ValueError(
            f"OOF coverage mismatch: missing={len(set(expected.tile)-set(predictions.tile))}, "
            f"extra={len(set(predictions.tile)-set(expected.tile))}"
        )
    rows = predictions.merge(
        labels[["image", "tile", "collection", "tissue", "injury"]],
        on=["image", "tile"], validate="one_to_one",
    )
    report = {
        "overall": evaluate(rows),
        "by_collection": {name: evaluate(group) for name, group in rows.groupby("collection")},
        "by_fold": {str(int(fold)): evaluate(group) for fold, group in rows.groupby("fold")},
    }
    output = args.output or args.predictions.with_name(args.predictions.stem + "_evaluation.json")
    json_dump(report, output)
    print(f"Wrote {output}")
    print(report["overall"])


if __name__ == "__main__":
    main()
