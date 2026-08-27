from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .validate_data import validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("completed_scores", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    incoming = pd.read_csv(args.completed_scores, dtype=str).fillna("")
    if not {"image", "human_score"}.issubset(incoming.columns):
        raise ValueError("completed score file needs image,human_score columns")
    incoming = incoming[["image", "human_score"]]
    incoming = incoming[incoming.human_score.str.strip() != ""].copy()
    numeric = pd.to_numeric(incoming.human_score, errors="coerce")
    if numeric.isna().any() or not numeric.between(1, 5).all() or not np_all_integer(numeric):
        raise ValueError("every supplied human_score must be an integer from 1 through 5")
    incoming["human_score"] = numeric.astype(int)
    if incoming.image.duplicated().any():
        raise ValueError("incoming score file contains duplicate image names")

    target = root / "human_thrips_scores.csv"
    plants_target = root / "plants.csv"
    existing = pd.read_csv(target)
    overlap = existing.merge(incoming, on="image", suffixes=("_old", "_new"))
    conflicts = overlap[overlap.human_score_old.astype(int) != overlap.human_score_new.astype(int)]
    if len(conflicts):
        raise ValueError(f"refusing to overwrite existing scores: {conflicts.image.tolist()}")
    combined = pd.concat([existing, incoming[~incoming.image.isin(existing.image)]], ignore_index=True)
    combined = combined.sort_values("image").reset_index(drop=True)

    backup = root / "metadata" / "backups" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_scores")
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(target, backup / target.name)
    shutil.copy2(plants_target, backup / plants_target.name)
    temporary = target.with_suffix(".csv.tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(target)
    plants = pd.read_csv(plants_target, dtype=str).fillna("")
    score_map = combined.assign(human_score=combined.human_score.astype(str)).set_index("image")["human_score"]
    plants["human_score"] = plants["image"].map(score_map).fillna("")
    plants_temporary = plants_target.with_suffix(".csv.tmp")
    plants.to_csv(plants_temporary, index=False)
    plants_temporary.replace(plants_target)
    report = validate(root, require_all_scores=True)
    if not report["ok"]:
        shutil.copy2(backup / target.name, target)
        shutil.copy2(backup / plants_target.name, plants_target)
        raise ValueError("import failed validation and was rolled back: " + "; ".join(report["errors"]))
    print(f"Imported scores: {len(combined)} total. Backup: {backup}")


def np_all_integer(series: pd.Series) -> bool:
    return bool(((series % 1) == 0).all())


if __name__ == "__main__":
    main()
