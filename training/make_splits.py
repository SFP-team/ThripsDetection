from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import load_config, seed_everything
from .validate_data import validate


def severity_band(row: pd.Series) -> str:
    counts = np.array([row.n_healthy, row.n_mild, row.n_severe], dtype=float)
    return ["healthy", "mild", "severe"][int(counts.argmax())]


def choose_holdout(plants: pd.DataFrame, n_test: int, seed: int) -> np.ndarray:
    """Select an exact, deterministic holdout balanced by collection and score.

    At least two examples of every score are kept in development whenever that
    is mathematically possible. This matters more than perfectly mirroring a
    very rare endpoint in a 25-plant test set.
    """
    rng = np.random.default_rng(seed)
    strata = plants["collection"].astype(str) + "__" + plants["human_score"].astype(str)
    groups = {key: list(indexes) for key, indexes in plants.groupby(strata).groups.items()}
    for indexes in groups.values():
        rng.shuffle(indexes)
    score_total = plants["human_score"].value_counts().to_dict()
    selected: list[int] = []
    selected_by_group = {key: 0 for key in groups}
    selected_by_score = {key: 0 for key in score_total}
    ideal = {key: n_test * len(indexes) / len(plants) for key, indexes in groups.items()}
    tie_break = {key: float(rng.random()) for key in groups}
    while len(selected) < n_test:
        candidates = []
        for key, indexes in groups.items():
            if selected_by_group[key] >= len(indexes):
                continue
            score = plants.loc[indexes[0], "human_score"]
            max_for_score = max(0, int(score_total[score]) - 2)
            if selected_by_score[score] >= max_for_score:
                continue
            deficit = ideal[key] - selected_by_group[key]
            candidates.append((deficit, -selected_by_group[key], tie_break[key], key))
        if not candidates:
            raise ValueError(
                f"cannot reserve {n_test} test plants while keeping two development examples per score"
            )
        key = max(candidates)[-1]
        index = groups[key][selected_by_group[key]]
        selected.append(index)
        selected_by_group[key] += 1
        score = plants.loc[index, "human_score"]
        selected_by_score[score] += 1
    return np.asarray(selected, dtype=int)


def assign_balanced_folds(dev: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    """Greedily balance score and collection without requiring large strata."""
    rng = np.random.default_rng(seed)
    assignments = pd.Series(-1, index=dev.index, dtype=int)
    fold_sizes = np.zeros(n_folds, dtype=int)
    score_counts: dict[tuple[int, int], int] = {}
    collection_counts: dict[tuple[str, int], int] = {}
    groups = []
    for key, indexes in dev.groupby(["human_score", "collection"]).groups.items():
        indexes = list(indexes)
        rng.shuffle(indexes)
        groups.append((len(indexes), key, indexes))
    groups.sort(key=lambda item: item[0])  # place rare combinations first
    for _, (score, collection), indexes in groups:
        for index in indexes:
            order = list(range(n_folds))
            rng.shuffle(order)
            fold = min(
                order,
                key=lambda candidate: (
                    fold_sizes[candidate],
                    score_counts.get((int(score), candidate), 0),
                    collection_counts.get((str(collection), candidate), 0),
                ),
            )
            assignments.loc[index] = fold
            fold_sizes[fold] += 1
            score_counts[(int(score), fold)] = score_counts.get((int(score), fold), 0) + 1
            collection_counts[(str(collection), fold)] = collection_counts.get((str(collection), fold), 0) + 1
    return assignments


def make_splits(dataset_root: Path, output: Path) -> pd.DataFrame:
    cfg = load_config()
    report = validate(dataset_root, require_all_scores=True)
    if not report["ok"]:
        raise ValueError("dataset validation failed: " + "; ".join(report["errors"]))
    seed_everything(int(cfg["seed"]))

    plants = pd.read_csv(dataset_root / "plants.csv")
    scores = pd.read_csv(dataset_root / "human_thrips_scores.csv")
    plants = plants.drop(columns=["human_score"], errors="ignore").merge(scores, on="image", validate="one_to_one")
    plants["severity_band"] = plants.apply(severity_band, axis=1)
    n_test = int(cfg["final_test_plants"])
    test_idx = choose_holdout(plants, n_test, int(cfg["seed"]))
    plants["split"] = "dev"
    plants.loc[test_idx, "split"] = "test"
    plants["fold"] = -1

    dev = plants[plants.split == "dev"].copy()
    plants.loc[dev.index, "fold"] = assign_balanced_folds(
        dev, int(cfg["n_folds"]), int(cfg["seed"])
    )

    columns = ["image", "collection", "human_score", "severity_band", "split", "fold"]
    result = plants[columns].sort_values("image").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/splits.csv"))
    args = parser.parse_args()
    result = make_splits(args.dataset_root.resolve(), args.output.resolve())
    print(result.groupby(["split", "collection", "human_score"]).size().to_string())
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
