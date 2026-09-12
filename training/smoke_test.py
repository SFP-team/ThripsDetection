from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .common import load_config, seed_everything
from .data import DAMAGE_TO_ID, TileDataset, build_transform
from .model import ThripsTileModel, coral_targets, ordinal_probabilities


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--model-id", default=cfg["fallback_model_id"])
    parser.add_argument("--finetune", action="store_true", help="also unfreeze the final backbone stage")
    args = parser.parse_args()
    seed_everything(int(cfg["seed"]))
    root = args.dataset_root.resolve()
    rows = pd.read_csv(root / "training_tiles.csv")
    rows["smoke_stratum"] = rows.apply(
        lambda row: row.injury if row.tissue == "flush" else row.tissue, axis=1
    )
    rows = rows.groupby("smoke_stratum", group_keys=False).head(2).reset_index(drop=True)
    if args.finetune:
        rows = rows.groupby("smoke_stratum", group_keys=False).head(1).reset_index(drop=True)
    dataset = TileDataset(rows, root, build_transform(args.model_id, int(cfg["input_size"]), False))
    batch = [dataset[i] for i in range(len(dataset))]
    pixels = torch.stack([item["pixel_values"] for item in batch])
    tissue = torch.stack([item["tissue_target"] for item in batch])
    damage = torch.stack([item["damage_target"] for item in batch])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ThripsTileModel(args.model_id, dropout=float(cfg["dropout"]), local_files_only=True)
    model.freeze_backbone()
    selected = model.unfreeze_last_stage() if args.finetune else []
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=3e-4
    )
    output = model(pixels.to(device))
    tissue = tissue.to(device)
    damage = damage.to(device)
    valid = damage >= 0
    loss = F.binary_cross_entropy_with_logits(output["tissue_logit"], tissue)
    loss = loss + F.binary_cross_entropy_with_logits(
        output["damage_logits"][valid], coral_targets(damage[valid])
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    probabilities = ordinal_probabilities(output["damage_logits"]).detach()
    if not torch.isfinite(loss) or not torch.allclose(
        probabilities.sum(dim=1), torch.ones(len(probabilities), device=device), atol=1e-5
    ):
        raise RuntimeError("non-finite loss or invalid ordinal probabilities")
    print(
        f"PASS model={args.model_id} device={device} tiles={len(batch)} "
        f"loss={float(loss.detach()):.4f} output={tuple(probabilities.shape)} unfrozen={selected}"
    )


if __name__ == "__main__":
    main()
