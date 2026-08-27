from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def tile_metrics(
    tissue_true: np.ndarray,
    tissue_prob: np.ndarray,
    damage_true: np.ndarray,
    damage_prob: np.ndarray,
) -> dict:
    tissue_pred = (tissue_prob >= 0.5).astype(int)
    valid = damage_true >= 0
    damage_pred = damage_prob[valid].argmax(axis=1)
    labels = [0, 1, 2]
    precision, recall, f1, support = precision_recall_fscore_support(
        damage_true[valid], damage_pred, labels=labels, zero_division=0
    )
    return {
        "tissue_balanced_accuracy": float(balanced_accuracy_score(tissue_true, tissue_pred)),
        "tissue_f1": float(f1_score(tissue_true, tissue_pred, zero_division=0)),
        "damage_macro_f1": float(f1_score(damage_true[valid], damage_pred, average="macro", labels=labels)),
        "damage_balanced_accuracy": float(balanced_accuracy_score(damage_true[valid], damage_pred)),
        "damage_qwk": float(cohen_kappa_score(damage_true[valid], damage_pred, weights="quadratic")),
        "damage_per_class": {
            name: {"precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, name in enumerate(["healthy", "mild", "severe"])
        },
        "damage_confusion": confusion_matrix(damage_true[valid], damage_pred, labels=labels).tolist(),
    }

