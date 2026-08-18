#!/usr/bin/env python3
"""Calibrate classical thrips maps against the 36 human plant scores.

Leave-one-out is the honest number. In-sample fit is shown only so we
can see how much a 36-point line can overfit.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path("/tmp/thrips-cal")
OUT = Path("/Users/whiterose/Documents/PlantInfestation")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da and db else float("nan")


def ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def mae(a: list[float], b: list[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def clip(x: float, lo: float = 1.0, hi: float = 5.0) -> float:
    return max(lo, min(hi, x))


def qwk(y: list[int], p: list[int], minv: int = 1, maxv: int = 5) -> float:
    k = maxv - minv + 1
    obs = [[0] * k for _ in range(k)]
    for a, b in zip(y, p):
        obs[a - minv][b - minv] += 1
    n = len(y)
    wy = [sum(obs[i]) for i in range(k)]
    wp = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    exp = [[wy[i] * wp[j] / n for j in range(k)] for i in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * exp[i][j] for i in range(k) for j in range(k))
    return 1 - num / den if den else float("nan")


def fit_linear(x: list[list[float]], y: list[float]) -> list[float]:
    # columns: intercept + features
    n = len(y)
    p = len(x[0]) + 1
    a = [[1.0] + row[:] for row in x]
    ata = [[sum(a[i][r] * a[i][c] for i in range(n)) for c in range(p)] for r in range(p)]
    aty = [sum(a[i][r] * y[i] for i in range(n)) for r in range(p)]
    # Gaussian elimination
    m = [ata[r][:] + [aty[r]] for r in range(p)]
    for col in range(p):
        pivot = max(range(col, p), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            return [sum(y) / n] + [0.0] * (p - 1)
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(p):
            if r == col:
                continue
            factor = m[r][col]
            m[r] = [m[r][c] - factor * m[col][c] for c in range(p + 1)]
    return [m[r][p] for r in range(p)]


def predict_linear(coef: list[float], row: list[float]) -> float:
    return clip(coef[0] + sum(c * v for c, v in zip(coef[1:], row)))


def pava(y: list[float], w: list[float] | None = None) -> list[float]:
    if w is None:
        w = [1.0] * len(y)
    blocks = [[y[i], w[i], 1] for i in range(len(y))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] <= blocks[i + 1][0] + 1e-15:
            i += 1
            continue
        total_w = blocks[i][1] + blocks[i + 1][1]
        mean = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / total_w
        count = blocks[i][2] + blocks[i + 1][2]
        blocks[i : i + 2] = [[mean, total_w, count]]
        i = max(i - 1, 0)
    out: list[float] = []
    for mean, _w, count in blocks:
        out.extend([mean] * count)
    return out


def fit_isotonic(x: list[float], y: list[float]) -> tuple[list[float], list[float]]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in order]
    ys = pava([y[i] for i in order])
    # collapse unique x to average y
    uniq_x: list[float] = []
    uniq_y: list[float] = []
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        uniq_x.append(xs[i])
        uniq_y.append(sum(ys[i : j + 1]) / (j - i + 1))
        i = j + 1
    return uniq_x, uniq_y


def predict_isotonic(xs: list[float], ys: list[float], x: float) -> float:
    if x <= xs[0]:
        return clip(ys[0])
    if x >= xs[-1]:
        return clip(ys[-1])
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            if xs[i + 1] == xs[i]:
                return clip(ys[i])
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return clip(ys[i] + t * (ys[i + 1] - ys[i]))
    return clip(ys[-1])


def metrics(human: list[float], pred: list[float]) -> dict[str, float]:
    rounded = [int(round(clip(p))) for p in pred]
    truth = [int(h) for h in human]
    return {
        "pearson": round(pearson(human, pred), 3),
        "spearman": round(spearman(human, pred), 3),
        "mae": round(mae(human, pred), 3),
        "mae_rounded": round(mae(human, [float(p) for p in rounded]), 3),
        "exact": int(sum(h == p for h, p in zip(truth, rounded))),
        "off_by_1": int(sum(abs(h - p) <= 1 for h, p in zip(truth, rounded))),
        "qwk": round(qwk(truth, rounded), 3),
    }


def loocv_linear(x: list[list[float]], y: list[float]) -> list[float]:
    preds = []
    for i in range(len(y)):
        xt = x[:i] + x[i + 1 :]
        yt = y[:i] + y[i + 1 :]
        coef = fit_linear(xt, yt)
        preds.append(predict_linear(coef, x[i]))
    return preds


def loocv_isotonic(x: list[float], y: list[float]) -> list[float]:
    preds = []
    for i in range(len(y)):
        xs, ys = fit_isotonic(x[:i] + x[i + 1 :], y[:i] + y[i + 1 :])
        preds.append(predict_isotonic(xs, ys, x[i]))
    return preds


def mean_by_bin(human: list[float], pred: list[float]) -> list[float]:
    groups: dict[int, list[float]] = defaultdict(list)
    for h, p in zip(human, pred):
        groups[int(h)].append(p)
    return [round(sum(groups[k]) / len(groups[k]), 2) if groups[k] else 0.0 for k in range(1, 6)]


def main() -> None:
    human_rows = {r["image"]: int(r["human_score"]) for r in read_csv(ROOT / "human_thrips_scores.csv")}
    old = {r["file"]: r for r in read_csv(ROOT / "maps" / "scores.csv")}
    extra = {r["file"]: r for r in read_csv(ROOT / "score_check" / "human_vs_classical.csv")}
    metrics_rows = {r["file"]: r for r in read_csv(ROOT / "image_metrics.csv")}
    foliage = read_csv(ROOT / "birefnet_tiles" / "tiles_foliage.csv")
    kept = defaultdict(int)
    for row in foliage:
        if row["decision"] == "keep":
            kept[row["image"]] += 1

    names = sorted(human_rows)
    y = [float(human_rows[n]) for n in names]
    classical = [float(old[n]["score_0_5"]) for n in names]
    off = [float(old[n]["offgreen_frac"]) for n in names]
    dark = [float(old[n]["darkspot_frac"]) for n in names]
    cr = [float(old[n]["crumple_frac"]) for n in names]
    b_off = [float(extra[n]["birefnet_offgreen"]) for n in names]
    b_dark = [float(extra[n]["birefnet_darkspot"]) for n in names]
    sharp = [float(metrics_rows[n]["sharp_var"]) / 1000.0 for n in names]
    n_fol = [float(kept[n]) for n in names]

    mean_y = sum(y) / len(y)
    baseline = [mean_y] * len(y)

    models: dict[str, list[float]] = {}
    models["always_mean"] = baseline
    models["uncalibrated_map"] = classical

    coef_map = fit_linear([[v] for v in classical], y)
    models["linear_map_insample"] = [predict_linear(coef_map, [v]) for v in classical]
    models["linear_map_loocv"] = loocv_linear([[v] for v in classical], y)

    xs, ys = fit_isotonic(classical, y)
    models["isotonic_map_insample"] = [predict_isotonic(xs, ys, v) for v in classical]
    models["isotonic_map_loocv"] = loocv_isotonic(classical, y)

    models["linear_off_loocv"] = loocv_linear([[v] for v in off], y)
    models["linear_off_dark_loocv"] = loocv_linear([[a, b] for a, b in zip(off, dark)], y)
    models["linear_off_dark_cr_loocv"] = loocv_linear(
        [[a, b, c] for a, b, c in zip(off, dark, cr)], y
    )
    models["linear_birefnet_off_dark_loocv"] = loocv_linear(
        [[a, b] for a, b in zip(b_off, b_dark)], y
    )
    models["linear_off_dark_sharp_loocv"] = loocv_linear(
        [[a, b, s] for a, b, s in zip(off, dark, sharp)], y
    )
    models["linear_off_dark_tiles_loocv"] = loocv_linear(
        [[a, b, t] for a, b, t in zip(off, dark, n_fol)], y
    )

    # Best in-sample formula for deployment (fit on all 36)
    coef_off_dark = fit_linear([[a, b] for a, b in zip(off, dark)], y)
    models["linear_off_dark_insample"] = [
        predict_linear(coef_off_dark, [a, b]) for a, b in zip(off, dark)
    ]

    summary = {name: metrics(y, pred) for name, pred in models.items()}
    summary["_notes"] = {
        "human_mean": round(mean_y, 3),
        "linear_map_coef": [round(c, 4) for c in coef_map],
        "linear_off_dark_coef": [round(c, 4) for c in coef_off_dark],
        "formula_off_dark": (
            f"clip(1..5, {coef_off_dark[0]:.3f} + "
            f"{coef_off_dark[1]:.3f}*offgreen + {coef_off_dark[2]:.3f}*darkspot)"
        ),
        "mean_pred_by_human_loocv_off_dark": mean_by_bin(y, models["linear_off_dark_loocv"]),
        "mean_pred_by_human_uncalibrated": mean_by_bin(y, classical),
        "mean_pred_by_human_isotonic_loocv": mean_by_bin(y, models["isotonic_map_loocv"]),
    }

    pred_rows = []
    for i, name in enumerate(names):
        pred_rows.append(
            {
                "image": name,
                "human": int(y[i]),
                "uncalibrated_map": round(classical[i], 2),
                "linear_map_loocv": round(models["linear_map_loocv"][i], 2),
                "isotonic_map_loocv": round(models["isotonic_map_loocv"][i], 2),
                "off_dark_loocv": round(models["linear_off_dark_loocv"][i], 2),
                "off_dark_insample": round(models["linear_off_dark_insample"][i], 2),
                "off_dark_rounded": int(round(models["linear_off_dark_loocv"][i])),
            }
        )

    out_json = OUT / "calibration_summary.json"
    out_csv = OUT / "calibrated_predictions.csv"
    out_json.write_text(json.dumps(summary, indent=2) + "\n")
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pred_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pred_rows)

    print(json.dumps(summary, indent=2))
    worst = sorted(
        pred_rows,
        key=lambda r: -abs(r["human"] - r["off_dark_loocv"]),
    )[:8]
    print("worst off_dark_loocv")
    for row in worst:
        print(row)


if __name__ == "__main__":
    main()
