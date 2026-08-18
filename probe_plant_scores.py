#!/usr/bin/env python3
"""Frozen ImageNet probe on plant crops, leave-one-out vs the 36 human scores."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGES = Path("/home/fpt/Chili thrips detection pictures")
CROPS = IMAGES / "analysis_2026-08-17" / "birefnet_tiles" / "plant_crops"
HUMAN = IMAGES / "analysis_2026-08-17" / "human_thrips_scores.csv"
OUT = IMAGES / "analysis_2026-08-17" / "score_check"


def clip(x: float) -> float:
    return float(np.clip(x, 1.0, 5.0))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ranks(xs: np.ndarray) -> np.ndarray:
    order = np.argsort(xs)
    out = np.empty_like(xs, dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(ranks(a), ranks(b))


def qwk(y: np.ndarray, p: np.ndarray) -> float:
    y = y.astype(int)
    p = p.astype(int)
    k = 5
    obs = np.zeros((k, k))
    for a, b in zip(y, p):
        obs[a - 1, b - 1] += 1
    n = len(y)
    wy = obs.sum(axis=1)
    wp = obs.sum(axis=0)
    exp = np.outer(wy, wp) / n
    w = np.array([[((i - j) ** 2) / 16 for j in range(k)] for i in range(k)])
    den = float((w * exp).sum())
    return float(1 - (w * obs).sum() / den) if den else float("nan")


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rounded = np.clip(np.rint(pred), 1, 5).astype(int)
    return {
        "pearson": round(pearson(y, pred), 3),
        "spearman": round(spearman(y, pred), 3),
        "mae": round(float(np.mean(np.abs(y - pred))), 3),
        "mae_rounded": round(float(np.mean(np.abs(y - rounded))), 3),
        "exact": int(np.sum(y.astype(int) == rounded)),
        "off_by_1": int(np.sum(np.abs(y.astype(int) - rounded) <= 1)),
        "qwk": round(qwk(y, rounded), 3),
    }


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xt = np.concatenate([np.ones((len(x), 1)), x], axis=1)
    a = xt.T @ xt
    a[np.diag_indices_from(a)] += lam
    a[0, 0] -= lam
    return np.linalg.solve(a, xt.T @ y)


def ridge_pred(coef: np.ndarray, x: np.ndarray) -> float:
    return clip(float(coef[0] + coef[1:] @ x))


def loocv_ridge(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        mu = x[mask].mean(axis=0)
        sd = x[mask].std(axis=0)
        sd[sd < 1e-6] = 1.0
        xt = (x[mask] - mu) / sd
        xi = (x[i] - mu) / sd
        coef = ridge_fit(xt, y[mask], lam)
        preds[i] = ridge_pred(coef, xi)
    return preds


def load_human() -> dict[str, float]:
    rows = list(csv.DictReader(HUMAN.open()))
    return {r["image"]: float(r["human_score"]) for r in rows}


@torch.inference_mode()
def embeddings(paths: list[Path], device: torch.device) -> np.ndarray:
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = efficientnet_v2_s(weights=weights).to(device).eval()
    feats = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, [384, 384], interpolation=InterpolationMode.BILINEAR)
        tensor = TF.normalize(
            TF.to_tensor(image),
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ).unsqueeze(0).to(device)
        feat = model.features(tensor)
        feat = torch.nn.functional.adaptive_avg_pool2d(feat, 1).flatten(1)
        feats.append(feat.float().cpu().numpy()[0])
    return np.stack(feats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    human = load_human()
    names = sorted(human)
    y = np.array([human[n] for n in names], dtype=float)
    crop_paths = [CROPS / f"{Path(n).stem}_plant.jpg" for n in names]
    missing = [str(p) for p in crop_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing[:3])

    device = torch.device("cuda:0")
    x = embeddings(crop_paths, device)
    np.save(OUT / "plant_crop_effv2s.npy", x)

    results = {
        "always_mean": metrics(y, np.full_like(y, y.mean())),
    }
    preds = {}
    for lam in (1.0, 10.0, 50.0, 200.0, 1000.0):
        pred = loocv_ridge(x, y, lam)
        results[f"effv2s_ridge_{lam:g}_loocv"] = metrics(y, pred)
        preds[lam] = pred

    best_lam = min(preds, key=lambda lam: results[f"effv2s_ridge_{lam:g}_loocv"]["mae"])
    best = preds[best_lam]
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "image": name,
                "human": int(y[i]),
                "probe_loocv": round(float(best[i]), 2),
                "probe_rounded": int(np.clip(np.rint(best[i]), 1, 5)),
            }
        )
    with (OUT / "imagenet_probe_loocv.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "best_lambda": best_lam,
        "results": results,
        "mean_pred_by_human": [
            round(float(best[y == k].mean()), 2) if np.any(y == k) else None
            for k in range(1, 6)
        ],
    }
    (OUT / "imagenet_probe_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
