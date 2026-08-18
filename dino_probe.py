#!/usr/bin/env python3
"""DINOv2 embeddings of foliage tiles + plant crops, LOOCV vs 36 human scores.

Tests whether modern self-supervised features carry the severity signal
that ImageNet-supervised EfficientNet features did not.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

ROOT = Path("/home/fpt/Chili thrips detection pictures/analysis_2026-08-17")
RUN = ROOT / "birefnet_tiles"
OUT = ROOT / "score_check"

MODEL_CANDIDATES = [
    "vit_base_patch16_dinov3.lvd1689m",
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_base_patch14_dinov2.lvd142m",
]


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
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
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


def ridge_loocv(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    preds = np.zeros(len(y))
    for i in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[i] = False
        mu = x[mask].mean(axis=0)
        sd = x[mask].std(axis=0)
        sd[sd < 1e-6] = 1.0
        xt = (x[mask] - mu) / sd
        xi = (x[i] - mu) / sd
        xt1 = np.concatenate([np.ones((xt.shape[0], 1)), xt], axis=1)
        a = xt1.T @ xt1
        a[np.diag_indices_from(a)] += lam
        a[0, 0] -= lam
        coef = np.linalg.solve(a, xt1.T @ y[mask])
        preds[i] = clip(float(coef[0] + coef[1:] @ xi))
    return preds


@torch.inference_mode()
def embed(model, transform, paths: list[Path], device) -> np.ndarray:
    feats = []
    batch = []
    for path in paths:
        batch.append(transform(Image.open(path).convert("RGB")))
        if len(batch) == 16:
            x = torch.stack(batch).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats.append(model(x).float().cpu().numpy())
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feats.append(model(x).float().cpu().numpy())
    return np.concatenate(feats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")

    model = None
    model_name = None
    for name in MODEL_CANDIDATES:
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
            model_name = name
            break
        except Exception as exc:  # noqa: BLE001
            print(f"could not load {name}: {exc}", flush=True)
    if model is None:
        raise RuntimeError("no backbone loaded")
    model = model.to(device).eval()
    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    print("backbone:", model_name, "input:", cfg["input_size"], flush=True)

    human = {
        r["image"]: float(r["human_score"])
        for r in csv.DictReader((ROOT / "human_thrips_scores.csv").open())
    }
    names = sorted(human)
    y = np.array([human[n] for n in names])

    tile_rows = [
        r
        for r in csv.DictReader((RUN / "tiles_foliage.csv").open())
        if r["decision"] == "keep"
    ]
    by_plant: dict[str, list[dict]] = defaultdict(list)
    for r in tile_rows:
        by_plant[r["image"]].append(r)

    tile_paths = [RUN / "foliage_tiles" / r["tile"] for r in tile_rows]
    crop_paths = [RUN / "plant_crops" / f"{Path(n).stem}_plant.jpg" for n in names]

    print(f"embedding {len(tile_paths)} tiles + {len(crop_paths)} crops", flush=True)
    tile_emb = embed(model, transform, tile_paths, device)
    crop_emb = embed(model, transform, crop_paths, device)
    np.save(OUT / "dino_tile_emb.npy", tile_emb)
    np.save(OUT / "dino_crop_emb.npy", crop_emb)

    tile_index = {r["tile"]: i for i, r in enumerate(tile_rows)}
    reps: dict[str, np.ndarray] = {}
    mean_rep = []
    flush_rep = []
    for n in names:
        rows = by_plant[n]
        idx = [tile_index[r["tile"]] for r in rows]
        emb = tile_emb[idx]
        rel = np.array([float(r["rel_y"]) for r in rows])
        w = np.where(rel <= 0.45, 1.0, 0.2)
        mean_rep.append(emb.mean(axis=0))
        flush_rep.append((emb * w[:, None]).sum(axis=0) / w.sum())
    reps["crop"] = crop_emb
    reps["tile_mean"] = np.stack(mean_rep)
    reps["tile_flush"] = np.stack(flush_rep)
    reps["crop_plus_tilemean"] = np.concatenate([crop_emb, np.stack(mean_rep)], axis=1)

    results: dict[str, dict] = {
        "backbone": {"name": model_name},
        "always_mean": metrics(y, np.full_like(y, y.mean())),
    }
    best = None
    for rep_name, x in reps.items():
        for lam in (10.0, 100.0, 1000.0, 5000.0):
            pred = ridge_loocv(x, y, lam)
            key = f"{rep_name}_lam{lam:g}"
            m = metrics(y, pred)
            results[key] = m
            if best is None or m["mae"] < best[1]["mae"]:
                best = (key, m, pred)

    results["_best"] = {"key": best[0], **best[1]}
    results["_best_mean_pred_by_human"] = [
        round(float(best[2][y == k].mean()), 2) if np.any(y == k) else None
        for k in range(1, 6)
    ]
    (OUT / "dino_probe_summary.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
