from __future__ import annotations

import argparse
from pathlib import Path

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
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(int(cfg["n_folds"])))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stage", choices=["probe", "finetune"], default="finetune")
    parser.add_argument("--output", type=Path, default=Path("training/artifacts/oof_tiles.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    splits = pd.read_csv(args.splits)
    tiles = pd.read_csv(root / "training_tiles.csv").merge(
        splits[["image", "split", "fold"]], on="image", validate="many_to_one"
    )
    rows = tiles[(tiles.split == "dev") & (tiles.fold == args.fold)].copy()
    if args.checkpoint is None:
        candidates = list(Path("training/artifacts").glob(f"*/fold_{args.fold}/{args.stage}/best.pt"))
        if len(candidates) != 1:
            raise ValueError(f"expected one checkpoint for fold {args.fold}, found {candidates}")
        checkpoint_path = candidates[0]
    else:
        checkpoint_path = args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ThripsTileModel(checkpoint["model_id"], dropout=0.0, local_files_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ds = TileDataset(rows, root, build_transform(checkpoint["model_id"], int(cfg["input_size"]), False))
    loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=args.workers,
                        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0)
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["pixel_values"].to(device, non_blocking=True))
            p_flush = torch.sigmoid(output["tissue_logit"]).float().cpu().numpy()
            p_damage = ordinal_probabilities(output["damage_logits"]).float().cpu().numpy()
            for i, tile in enumerate(batch["tile"]):
                predictions.append({
                    "image": batch["image"][i], "tile": tile, "fold": args.fold,
                    "p_flush": p_flush[i], "p_healthy": p_damage[i, 0],
                    "p_mild": p_damage[i, 1], "p_severe": p_damage[i, 2],
                    "expected_damage": p_damage[i, 1] + 2 * p_damage[i, 2],
                })
    new = pd.DataFrame(predictions)
    if args.output.exists():
        old = pd.read_csv(args.output)
        old = old[old.fold != args.fold]
        new = pd.concat([old, new], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    new.sort_values(["fold", "image", "tile"]).to_csv(args.output, index=False)
    print(f"Wrote {len(predictions)} fold-{args.fold} predictions to {args.output}")


if __name__ == "__main__":
    main()

