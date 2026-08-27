from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from .common import json_dump, load_config, seed_everything
from .data import TileDataset, build_transform, plant_class_sampler
from .metrics import tile_metrics
from .model import ThripsTileModel, coral_targets, ordinal_probabilities


def collate(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "tissue_target": torch.stack([x["tissue_target"] for x in batch]),
        "damage_target": torch.stack([x["damage_target"] for x in batch]),
        "tile": [x["tile"] for x in batch],
        "image": [x["image"] for x in batch],
    }


def run_epoch(model, loader, device, optimizer=None, accumulation: int = 1) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    # A frozen feature probe must keep backbone dropout/drop-path and batch
    # statistics deterministic even while its small heads are training.
    if not any(parameter.requires_grad for parameter in model.backbone.parameters()):
        model.backbone.eval()
    losses: list[float] = []
    t_true: list[np.ndarray] = []
    t_prob: list[np.ndarray] = []
    d_true: list[np.ndarray] = []
    d_prob: list[np.ndarray] = []
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        tissue = batch["tissue_target"].to(device)
        damage = batch["damage_target"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(pixels)
            tissue_loss = F.binary_cross_entropy_with_logits(output["tissue_logit"], tissue)
            valid = damage >= 0
            damage_loss = (
                F.binary_cross_entropy_with_logits(output["damage_logits"][valid], coral_targets(damage[valid]))
                if valid.any()
                else output["damage_logits"].sum() * 0.0
            )
            loss = tissue_loss + damage_loss
        if training:
            (loss / accumulation).backward()
            if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().cpu()))
        t_true.append(tissue.detach().cpu().numpy())
        t_prob.append(torch.sigmoid(output["tissue_logit"]).detach().float().cpu().numpy())
        d_true.append(damage.detach().cpu().numpy())
        d_prob.append(ordinal_probabilities(output["damage_logits"]).detach().float().cpu().numpy())
    metrics = tile_metrics(
        np.concatenate(t_true), np.concatenate(t_prob), np.concatenate(d_true), np.concatenate(d_prob)
    )
    return float(np.mean(losses)), metrics


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True, choices=range(int(cfg["n_folds"])))
    parser.add_argument("--stage", choices=["probe", "finetune"], default="probe")
    parser.add_argument("--model-id", default=cfg["primary_model_id"])
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("training/artifacts"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    seed_everything(int(cfg["seed"]) + args.fold)
    root = args.dataset_root.resolve()
    splits = pd.read_csv(args.splits)
    tiles = pd.read_csv(root / "training_tiles.csv")
    tiles = tiles.merge(splits[["image", "split", "fold"]], on="image", validate="many_to_one")
    train_rows = tiles[(tiles.split == "dev") & (tiles.fold != args.fold)].copy()
    val_rows = tiles[(tiles.split == "dev") & (tiles.fold == args.fold)].copy()
    if args.smoke:
        train_rows = train_rows.groupby("tissue", group_keys=False).head(4)
        val_rows = val_rows.groupby("tissue", group_keys=False).head(4)
    if train_rows.empty or val_rows.empty:
        raise ValueError("empty train or validation fold")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ThripsTileModel(args.model_id, dropout=float(cfg["dropout"]))
    if args.stage == "finetune":
        init_checkpoint = args.init_checkpoint
        if init_checkpoint is None:
            init_checkpoint = (
                args.output_root / args.model_id.replace("/", "__") /
                f"fold_{args.fold}" / "probe" / "best.pt"
            )
        payload = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        if payload["model_id"] != args.model_id:
            raise ValueError(
                f"initial checkpoint model {payload['model_id']} does not match {args.model_id}"
            )
        model.load_state_dict(payload["state_dict"])
        print(f"Initialized from {init_checkpoint}")
    model.freeze_backbone()
    if args.stage == "finetune":
        selected = model.unfreeze_last_stage()
        print(f"Unfrozen: {selected}")
    model.to(device)

    train_ds = TileDataset(train_rows, root, build_transform(args.model_id, int(cfg["input_size"]), True))
    val_ds = TileDataset(val_rows, root, build_transform(args.model_id, int(cfg["input_size"]), False))
    sampler = plant_class_sampler(train_rows, int(cfg["seed"]) + args.fold)
    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["batch_size"]), sampler=sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=args.workers,
        pin_memory=True, collate_fn=collate, persistent_workers=args.workers > 0,
    )

    head_params = list(model.norm.parameters()) + list(model.tissue_head.parameters()) + list(model.damage_head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    groups = [{"params": head_params, "lr": float(cfg["head_learning_rate"])}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": float(cfg["backbone_learning_rate"])})
    optimizer = AdamW(groups, weight_decay=float(cfg["weight_decay"]))
    epochs = 1 if args.smoke else int(cfg["probe_epochs"] if args.stage == "probe" else cfg["finetune_epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    accumulation = int(cfg["gradient_accumulation"])

    out = args.output_root / args.model_id.replace("/", "__") / f"fold_{args.fold}" / args.stage
    out.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    history = []
    for epoch in range(epochs):
        started = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, device, optimizer, accumulation)
        with torch.inference_mode():
            val_loss, val_metrics = run_epoch(model, val_loader, device)
        scheduler.step()
        record = {
            "epoch": epoch + 1, "seconds": round(time.time() - started, 2),
            "train_loss": train_loss, "val_loss": val_loss,
            "train": train_metrics, "val": val_metrics,
        }
        history.append(record)
        score = val_metrics["damage_macro_f1"]
        print(f"epoch={epoch+1} val_loss={val_loss:.4f} macro_f1={score:.4f}")
        if score > best:
            best = score
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(), "model_id": args.model_id,
                    "fold": args.fold, "stage": args.stage, "config": cfg,
                    "val_metrics": val_metrics,
                },
                out / "best.pt",
            )
        else:
            stale += 1
        json_dump(history, out / "history.json")
        if not args.smoke and stale >= int(cfg["early_stopping_patience"]):
            break
    print(f"BEST damage_macro_f1={best:.4f} checkpoint={out / 'best.pt'}")


if __name__ == "__main__":
    main()
