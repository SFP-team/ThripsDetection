#!/usr/bin/env python3
"""Segment rover photos with Center BiRefNet fold 0, then tile the plant box.

The paper script scripts/benchmark_center_birefnet_20.py is locked to 20
named W20* images. This reuses the same fold-0 checkpoints and threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
TILE = 512
STRIDE = 384
MIN_MASK_FRAC = 0.35
BBOX_PAD = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def list_images(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_upright(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def stem_seed(rgb: np.ndarray, project: Path, width: int, height: int) -> tuple[float, float, str]:
    sam3 = project / "sam3_pipeline_original"
    if sam3.is_dir():
        sys.path.insert(0, str(sam3.resolve()))
        try:
            from SAM3_new.config import Sam3NewConfig
            from SAM3_new.stem_anchor import find_stem_seed

            x, y = find_stem_seed(rgb, Sam3NewConfig()).foreground_xy
            return float(x), float(y), "stem_anchor"
        except Exception as exc:  # noqa: BLE001
            print(f"stem seed failed ({exc}); using image center", flush=True)
    return width / 2.0, height / 2.0, "image_center"


def prepare_tensor(
    image: Image.Image,
    seed_x: float,
    seed_y: float,
    height: int,
    width: int,
    center_heatmap,
) -> torch.Tensor:
    original_width, original_height = image.size
    resized = TF.resize(
        image,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
    )
    rgb = TF.normalize(
        TF.to_tensor(resized),
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    )
    prompt = center_heatmap(
        seed_x,
        seed_y,
        original_width,
        original_height,
        width,
        height,
    )
    return torch.cat([rgb, prompt], dim=0).unsqueeze(0)


def resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def keep_largest(mask: np.ndarray) -> np.ndarray:
    from cv2 import CC_STAT_AREA, connectedComponentsWithStats

    if not mask.any():
        return mask
    count, labels, stats, _ = connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    areas = stats[1:, CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return labels == keep


def bbox(mask: np.ndarray, pad: int) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise RuntimeError("empty mask")
    height, width = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + 1 + pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    origins = list(range(0, length - tile + 1, stride))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    fg = mask.astype(bool)
    out[fg, 0] = (out[fg, 0].astype(np.float32) * 0.42).astype(np.uint8)
    out[fg, 1] = np.clip(out[fg, 1].astype(np.float32) * 0.42 + 148, 0, 255).astype(np.uint8)
    out[fg, 2] = (out[fg, 2].astype(np.float32) * 0.42).astype(np.uint8)
    edge = np.zeros_like(fg)
    edge[1:] |= fg[1:] != fg[:-1]
    edge[:-1] |= fg[:-1] != fg[1:]
    edge[:, 1:] |= fg[:, 1:] != fg[:, :-1]
    edge[:, :-1] |= fg[:, :-1] != fg[:, 1:]
    out[edge] = (255, 210, 0)
    return out


def contact_sheet(paths: list[Path], columns: int = 6, cell: int = 192) -> Image.Image:
    if not paths:
        return Image.new("RGB", (cell, cell), (20, 20, 20))
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), (18, 18, 18))
    for index, path in enumerate(paths):
        with Image.open(path) as tile:
            thumb = ImageOps.fit(tile.convert("RGB"), (cell, cell))
        x = (index % columns) * cell
        y = (index // columns) * cell
        sheet.paste(thumb, (x, y))
    return sheet


def main() -> None:
    args = parse_args()
    project = Path(args.project).resolve()
    sys.path.insert(0, str(project / "src"))
    from plant_benchmark.center_birefnet import center_heatmap, expand_center_channel
    from plant_benchmark.config import load_config
    from plant_benchmark.models import create_dense_model

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.cuda.set_device(device)

    cfg = load_config(project / args.config)
    images = list_images(Path(args.images).resolve())
    if not images:
        raise RuntimeError(f"No images in {args.images}")

    threshold_path = (
        Path(cfg["paths"]["runs"])
        / "birefnet_hr_centered"
        / "finetuned"
        / f"fold_{args.fold}"
        / "threshold.json"
    )
    threshold_payload = json.loads(threshold_path.read_text())
    threshold = float(threshold_payload["selected"])
    balanced_threshold = float(threshold_payload["balanced"]["threshold"])
    model_height = int(cfg["data"]["height"])
    model_width = int(cfg["data"]["width"])
    checkpoint = (
        Path(cfg["paths"]["runs"])
        / "birefnet_hr_centered"
        / "finetuned"
        / f"fold_{args.fold}"
        / "best.pt"
    )

    output = Path(args.output).resolve()
    mask_dir = output / "masks"
    overlay_dir = output / "overlays"
    overlay_review_dir = output / "overlays_review"
    crop_dir = output / "plant_crops"
    tile_dir = output / "tiles"
    sheet_dir = output / "contact_sheets"
    for folder in (mask_dir, overlay_dir, overlay_review_dir, crop_dir, tile_dir, sheet_dir):
        folder.mkdir(parents=True, exist_ok=True)

    # Same loader as scripts/benchmark_center_birefnet_20.py. create_center_model
    # refuses this fold-0 pair because the delta has no stored base SHA.
    model = create_dense_model("birefnet_hr", cfg)
    base_ckpt = Path(cfg["paths"]["runs"]) / "birefnet_hr" / "finetuned" / f"fold_{args.fold}" / "best.pt"
    base_payload = torch.load(base_ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(base_payload["model"])
    expand_center_channel(model)
    center_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = center_payload["trainable_state"]
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected Center BiRefNet tensors: {unexpected[:5]}")
    model = model.float().to(device).eval()
    rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    with torch.inference_mode():
        for index, path in enumerate(images, start=1):
            image = load_upright(path)
            width, height = image.size
            rgb = np.asarray(image)
            seed_x, seed_y, seed_method = stem_seed(rgb, project, width, height)
            tensor = prepare_tensor(
                image, seed_x, seed_y, model_height, model_width, center_heatmap
            ).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probability = model(tensor).sigmoid()[0, 0].float().cpu().numpy()
            del tensor

            selected = keep_largest(resize_mask(probability >= threshold, width, height))
            balanced = keep_largest(
                resize_mask(probability >= balanced_threshold, width, height)
            )
            mask = selected if selected.mean() >= 0.01 else balanced
            mask_name = "selected" if mask is selected else "balanced_fallback"

            Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                mask_dir / f"{path.stem}.mask.png"
            )

            x0, y0, x1, y1 = bbox(mask, BBOX_PAD)
            box_w, box_h = x1 - x0, y1 - y0
            plant_crop = image.crop((x0, y0, x1, y1))
            plant_crop.save(crop_dir / f"{path.stem}_plant.jpg", quality=90)

            painted = overlay(rgb, mask)
            drawn = Image.fromarray(painted)
            draw = ImageDraw.Draw(drawn)
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 220, 255), width=6)

            kept_paths: list[Path] = []
            xs = tile_origins(box_w, TILE, STRIDE)
            ys = tile_origins(box_h, TILE, STRIDE)
            tile_index = 0
            generated = 0
            for dy in ys:
                for dx in xs:
                    generated += 1
                    px = x0 + dx
                    py = y0 + dy
                    # If the plant box is smaller than TILE, take the box itself.
                    tw = min(TILE, box_w)
                    th = min(TILE, box_h)
                    if box_w < TILE:
                        px = x0
                    if box_h < TILE:
                        py = y0
                    tile_rgb = image.crop((px, py, px + tw, py + th))
                    if tile_rgb.size != (TILE, TILE):
                        tile_rgb = tile_rgb.resize((TILE, TILE), Image.Resampling.BILINEAR)
                        tile_mask = np.asarray(
                            Image.fromarray(mask[py : py + th, px : px + tw].astype(np.uint8) * 255)
                            .resize((TILE, TILE), Image.Resampling.NEAREST)
                        ) > 127
                    else:
                        tile_mask = mask[py : py + th, px : px + tw]
                    frac = float(tile_mask.mean())
                    keep = frac >= MIN_MASK_FRAC
                    color = (0, 255, 90) if keep else (180, 180, 180)
                    draw.rectangle((px, py, px + tw - 1, py + th - 1), outline=color, width=3)
                    if not keep:
                        continue
                    tile_index += 1
                    tile_name = f"{path.stem}_x{px}_y{py}.jpg"
                    tile_path = tile_dir / tile_name
                    tile_rgb.save(tile_path, quality=92)
                    kept_paths.append(tile_path)
                    rows.append(
                        {
                            "image": path.name,
                            "tile": tile_name,
                            "x": px,
                            "y": py,
                            "width": tw,
                            "height": th,
                            "mask_frac": round(frac, 4),
                            "seed_method": seed_method,
                            "mask_used": mask_name,
                        }
                    )

            drawn.save(overlay_dir / f"{path.stem}_overlay.jpg", quality=90)
            review = drawn.copy()
            review.thumbnail((1600, 1600))
            review.save(overlay_review_dir / f"{path.stem}_overlay.jpg", quality=85)
            contact_sheet(kept_paths).save(sheet_dir / f"{path.stem}_tiles.jpg", quality=85)

            image_rows.append(
                {
                    "image": path.name,
                    "width": width,
                    "height": height,
                    "seed_x": round(seed_x, 1),
                    "seed_y": round(seed_y, 1),
                    "seed_method": seed_method,
                    "mask_used": mask_name,
                    "selected_frac": round(float(selected.mean()), 4),
                    "balanced_frac": round(float(balanced.mean()), 4),
                    "box_x0": x0,
                    "box_y0": y0,
                    "box_x1": x1,
                    "box_y1": y1,
                    "box_w": box_w,
                    "box_h": box_h,
                    "tiles_generated": generated,
                    "tiles_kept": tile_index,
                }
            )
            print(
                f"[{index:02d}/{len(images)}] {path.name}: "
                f"box={box_w}x{box_h} kept={tile_index}/{generated} "
                f"mask={mask_name} seed={seed_method}",
                flush=True,
            )

    tiles_csv = output / "tiles.csv"
    with tiles_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)
    images_csv = output / "images.csv"
    with images_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    kept = [int(row["tiles_kept"]) for row in image_rows]
    summary = {
        "images": len(image_rows),
        "tiles_kept": len(rows),
        "tiles_kept_per_image_mean": round(float(np.mean(kept)), 2),
        "tiles_kept_per_image_min": int(min(kept)),
        "tiles_kept_per_image_max": int(max(kept)),
        "tile": TILE,
        "stride": STRIDE,
        "min_mask_frac": MIN_MASK_FRAC,
        "fold": args.fold,
        "threshold_selected": threshold,
        "threshold_balanced": balanced_threshold,
        "device": str(device),
        "checkpoint": str(checkpoint),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
