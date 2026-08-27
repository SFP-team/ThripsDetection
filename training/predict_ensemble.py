from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import load_config
from .data import TileDataset, build_transform
from .model import ThripsTileModel, ordinal_probabilities
from .train_tiles import collate


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path, default=Path("training/artifacts/splits.csv"))
    parser.add_argument(
        "--input-csv", type=Path,
        help="Optional tile manifest for an unlabeled run. Defaults to the locked final-test tiles.",
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--stage", choices=["probe", "finetune"], default="finetune")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if args.input_csv:
        rows = pd.read_csv(args.input_csv).copy()
        # Native segment_and_tile.py/filter_tube_tiles.py output has a
        # keep/drop decision and stores retained images under foliage_tiles/.
        # Normalize it here so UI jobs do not need a second manifest adapter.
        if "decision" in rows.columns:
            rows = rows[rows.decision.eq("keep")].copy()
        if "tile_path" not in rows.columns and "tile" in rows.columns:
            rows["tile_path"] = rows.tile.map(lambda value: f"foliage_tiles/{value}")
        if "collection" not in rows.columns:
            rows["collection"] = "field_upload"
    else:
        splits = pd.read_csv(args.splits)
        tiles = pd.read_csv(root / "training_tiles.csv").merge(
            splits[["image", "split"]], on="image", validate="many_to_one"
        )
        rows = tiles[tiles.split.eq("test")].copy()
    required = {"image", "tile", "tile_path", "x", "y", "rel_y"}
    missing_columns = required - set(rows.columns)
    if missing_columns:
        raise ValueError(f"tile manifest is missing {sorted(missing_columns)}")
    rows = rows.sort_values(["image", "tile"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("the requested tile set is empty")

    checkpoints = [args.checkpoint_root / f"fold_{fold}" / args.stage / "best.pt"
                   for fold in range(int(cfg["n_folds"]))]
    missing = [path for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing ensemble checkpoints: {missing}")

    first = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
    model_id = str(first["model_id"])
    dataset = TileDataset(rows, root, build_transform(model_id, int(cfg["input_size"]), False))
    loader = DataLoader(
        dataset, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tissue_sum = np.zeros(len(rows), dtype=np.float64)
    damage_sum = np.zeros((len(rows), 3), dtype=np.float64)

    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["model_id"] != model_id:
            raise ValueError(f"model mismatch in {checkpoint_path}")
        model = ThripsTileModel(model_id, dropout=0.0, local_files_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        offset = 0
        with torch.inference_mode():
            for batch in loader:
                output = model(batch["pixel_values"].to(device, non_blocking=True))
                tissue = torch.sigmoid(output["tissue_logit"]).float().cpu().numpy()
                damage = ordinal_probabilities(output["damage_logits"]).float().cpu().numpy()
                stop = offset + len(tissue)
                tissue_sum[offset:stop] += tissue
                damage_sum[offset:stop] += damage
                offset = stop
        if offset != len(rows):
            raise RuntimeError(f"checkpoint {checkpoint_path} predicted {offset} of {len(rows)} tiles")
        print(f"Predicted {len(rows)} tiles with {checkpoint_path}")
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tissue = tissue_sum / len(checkpoints)
    damage = damage_sum / len(checkpoints)
    output = rows[["image", "tile"]].copy()
    output["fold"] = -1
    output["p_flush"] = tissue
    output["p_healthy"] = damage[:, 0]
    output["p_mild"] = damage[:, 1]
    output["p_severe"] = damage[:, 2]
    output["expected_damage"] = damage[:, 1] + 2.0 * damage[:, 2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Wrote five-fold ensemble predictions for {len(output)} tiles to {args.output}")


if __name__ == "__main__":
    main()
