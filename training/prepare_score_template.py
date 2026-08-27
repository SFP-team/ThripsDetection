from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("training/artifacts/plant_scores_to_complete.csv")
    )
    args = parser.parse_args()
    plants = pd.read_csv(args.dataset_root / "plants.csv", dtype=str).fillna("")
    scores = pd.read_csv(args.dataset_root / "human_thrips_scores.csv", dtype=str).fillna("")
    scored = set(scores.image)
    columns = [
        "image", "collection", "human_score", "n_tiles", "n_flush",
        "n_healthy", "n_mild", "n_severe", "n_uncertain", "photo_path", "crop_path",
    ]
    missing = plants[~plants.image.isin(scored)].copy()
    missing["human_score"] = ""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    missing[columns].sort_values(["collection", "image"]).to_csv(args.output, index=False)
    print(f"Wrote {len(missing)} missing-score rows to {args.output}")


if __name__ == "__main__":
    main()

