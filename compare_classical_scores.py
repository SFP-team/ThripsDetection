#!/usr/bin/env python3
"""Recompute classical maps on BiRefNet foliage masks and compare to human 0-5."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

HUMAN = [
    5, 3, 4, 4, 3, 5, 4, 4, 2, 4, 2, 2, 2, 3, 1, 2,
    4, 3, 3, 3, 4, 4, 2, 3, 4, 2, 3, 1, 2, 2, 1, 3, 3, 3, 4, 3,
]
SRC = Path("/home/fpt/Chili thrips detection pictures")
RUN = SRC / "analysis_2026-08-17" / "birefnet_tiles"
OUT = SRC / "analysis_2026-08-17" / "score_check"


def offgreen_map(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, bb = cv2.split(lab)
    m = mask > 0
    if m.sum() < 50:
        return 0.0
    greenness = (-a) + 0.3 * bb
    thr = np.median(greenness[m])
    healthy = m & (greenness >= thr)
    proto = np.array(
        [np.median(L[healthy]), np.median(a[healthy]), np.median(bb[healthy])]
    )
    dist = np.sqrt((L - proto[0]) ** 2 + (a - proto[1]) ** 2 + (bb - proto[2]) ** 2)
    p95 = np.percentile(dist[m], 95) + 1e-6
    norm = np.clip(dist / p95, 0, 1)
    return float((norm[m] > 0.45).mean())


def darkspot_frac(bgr, mask):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    brown = ((h < 25) | (h > 170)) & (s > 40) & (v < 140) & (v > 20)
    dark = (v < 70) & (s < 80)
    spots = ((brown | dark) & (mask > 0)).astype(np.uint8) * 255
    spots = cv2.morphologyEx(spots, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(spots)
    clean = np.zeros_like(spots)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 20:
            clean[lab == i] = 255
    return float((clean > 0).sum() / max((mask > 0).sum(), 1))


def crumple_frac(bgr, mask):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.where(mask > 0, cv2.magnitude(gx, gy), 0)
    local = cv2.blur(mag, (21, 21))
    m = mask > 0
    if m.sum() < 50:
        return 0.0
    p95 = np.percentile(local[m], 95) + 1e-6
    norm = np.clip(local / p95, 0, 1)
    return float((norm[m] > 0.55).mean())


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    def ranks(xs):
        order = np.argsort(xs)
        ranks_out = np.empty(len(xs), float)
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            ranks_out[order[i : j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return ranks_out

    return pearson(ranks(np.asarray(a, float)), ranks(np.asarray(b, float)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = {r["file"]: r for r in csv.DictReader((SRC / "analysis_2026-08-17" / "maps" / "scores.csv").open())}
    images = sorted(SRC.glob("IMG_*.JPG"))
    rows = []
    for human, path in zip(HUMAN, images):
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError(path)
        # Match EXIF-upright foliage masks (portrait).
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageOps

        upright = np.asarray(ImageOps.exif_transpose(Image.fromarray(rgb)))
        bgr_u = cv2.cvtColor(upright, cv2.COLOR_RGB2BGR)
        mask_path = RUN / "masks" / f"{path.stem}.foliage.png"
        if not mask_path.is_file():
            mask_path = RUN / "masks" / f"{path.stem}.mask.png"
        mask = np.asarray(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != bgr_u.shape[:2]:
            mask = cv2.resize(mask, (bgr_u.shape[1], bgr_u.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8) * 255
        off = offgreen_map(bgr_u, mask)
        dark = darkspot_frac(bgr_u, mask)
        cr = crumple_frac(bgr_u, mask)
        raw = float(np.clip((2.2 * off + 3.5 * dark + 1.8 * cr) * 5.0, 0, 5))
        prev = old.get(path.name, {})
        rows.append(
            {
                "file": path.name,
                "human": human,
                "classical_rescaled": float(prev.get("score_0_5", "nan")),
                "offgreen_old": float(prev.get("offgreen_frac", "nan")),
                "birefnet_offgreen": round(off, 4),
                "birefnet_darkspot": round(dark, 4),
                "birefnet_crumple": round(cr, 4),
                "birefnet_raw_clip5": round(raw, 2),
                "foliage_pixels": int((mask > 0).sum()),
            }
        )
        print(path.name, human, round(off, 3), round(raw, 2), flush=True)

    with (OUT / "human_vs_classical.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    human = [r["human"] for r in rows]
    summary = {
        "n": 36,
        "classical_rescaled_pearson": round(pearson(human, [r["classical_rescaled"] for r in rows]), 3),
        "classical_rescaled_spearman": round(spearman(human, [r["classical_rescaled"] for r in rows]), 3),
        "birefnet_offgreen_pearson": round(pearson(human, [r["birefnet_offgreen"] for r in rows]), 3),
        "birefnet_offgreen_spearman": round(spearman(human, [r["birefnet_offgreen"] for r in rows]), 3),
        "birefnet_dark_pearson": round(pearson(human, [r["birefnet_darkspot"] for r in rows]), 3),
        "birefnet_crumple_pearson": round(pearson(human, [r["birefnet_crumple"] for r in rows]), 3),
        "birefnet_raw_pearson": round(pearson(human, [r["birefnet_raw_clip5"] for r in rows]), 3),
    }
    (OUT / "correlation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
