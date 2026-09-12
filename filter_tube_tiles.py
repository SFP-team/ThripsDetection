#!/usr/bin/env python3
"""Drop grow-tube and mulch tiles from the BiRefNet plant-tile set.

BiRefNet paints the white tube as foreground. This subtracts bright,
low-saturation pixels from each mask in the lower part of the plant box,
then keeps a tile only if enough of the remaining mask is foliage.

Foliage is excess-green anywhere, plus every plant-mask pixel in the upper
part of the box. Silvered or bronzed young flush is pale, fails the green
test, and sits at the top; the tube never sits there. Dropping those tiles
would drop the damage we are trying to score.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

MIN_FOLIAGE_MASK = 0.30
MIN_VEG_FRAC = 0.35
# Sky behind tips is bright and unsaturated, same as the tube.
# Only treat white as plastic in the lower part of the plant box.
TUBE_Y_FRAC = 0.55
TUBE_WHITE = 0.45
TUBE_VEG = 0.30
OUT_FIELDS = [
    "image", "tile", "x", "y", "width", "height", "mask_frac", "seed_method", "mask_used",
    "foliage_mask_frac", "veg_frac", "upper_foliage_frac", "white_frac", "rel_y", "decision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--run", required=True, help="birefnet_tiles output folder")
    parser.add_argument(
        "--skip-previews",
        action="store_true",
        help="Skip foliage overlays and contact sheets. Keep/drop copies and tiles_foliage.csv are still written.",
    )
    return parser.parse_args()


def load_upright(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def white_tube(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    red, green, blue = x[:, :, 0], x[:, :, 1], x[:, :, 2]
    maximum = x.max(axis=2)
    minimum = x.min(axis=2)
    value = maximum / 255.0
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 1)
    return (value > 0.72) & (saturation < 0.22)


def vegetation(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float32)
    excess_green = (2.0 * x[:, :, 1] - x[:, :, 0] - x[:, :, 2]) / 255.0
    return excess_green > 0.08


def contact_sheet(paths: list[Path], columns: int = 6, cell: int = 192) -> Image.Image:
    if not paths:
        return Image.new("RGB", (cell, cell), (20, 20, 20))
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * cell), (18, 18, 18))
    for index, path in enumerate(paths):
        with Image.open(path) as tile:
            thumb = ImageOps.fit(tile.convert("RGB"), (cell, cell))
        sheet.paste(thumb, ((index % columns) * cell, (index // columns) * cell))
    return sheet


def resize_bool(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
    ) > 127


def tile_patch(mask: np.ndarray, x: int, y: int, width: int, height: int, shape: tuple[int, int]) -> np.ndarray:
    """Mask pixels under one tile, resized to the tile image when the box was upsampled."""
    patch = mask[y : y + height, x : x + width]
    if patch.size == 0:
        return np.zeros(shape, dtype=bool)
    if patch.shape != shape:
        patch = resize_bool(patch, shape[1], shape[0])
    return patch


def main() -> None:
    args = parse_args()
    run = Path(args.run).resolve()
    image_dir = Path(args.images).resolve()
    tile_dir = run / "tiles"
    mask_dir = run / "masks"
    foliage_dir = run / "foliage_tiles"
    dropped_dir = run / "dropped_tiles"
    sheet_dir = run / "foliage_contact_sheets"
    overlay_dir = run / "foliage_overlays_review"
    folders = [foliage_dir, dropped_dir]
    if not args.skip_previews:
        folders.extend((sheet_dir, overlay_dir))
    for folder in folders:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    images = {row["image"]: row for row in csv.DictReader((run / "images.csv").open())}
    tiles = list(csv.DictReader((run / "tiles.csv").open()))
    out_rows: list[dict[str, object]] = []
    kept_by_image: dict[str, list[Path]] = {name: [] for name in images}
    tube_start_by_image: dict[str, int] = {}

    for name, info in images.items():
        rgb = load_upright(image_dir / name)
        plant = np.asarray(Image.open(mask_dir / f"{Path(name).stem}.mask.png")) > 127
        if plant.shape[:2] != rgb.shape[:2]:
            plant = resize_bool(plant, rgb.shape[1], rgb.shape[0])
        height = rgb.shape[0]
        ys = np.nonzero(plant)[0]
        y0 = int(ys.min()) if len(ys) else 0
        y1 = int(ys.max()) + 1 if len(ys) else height
        tube_start = y0 + int(TUBE_Y_FRAC * max(1, y1 - y0))
        tube_start_by_image[name] = tube_start
        lower = np.zeros(plant.shape, dtype=bool)
        lower[tube_start:] = True
        white = white_tube(rgb)
        tube = plant & white & lower
        # Green pixels count everywhere except on the tube (old rule). Above the
        # tube line, every non-white plant pixel counts too: silvered and bronzed
        # young flush fails the green test but sits at the top, where the only
        # bright unsaturated thing is plastic or sky.
        upper = plant & ~lower & ~white
        foliage_mask = (plant & vegetation(rgb) & ~tube) | upper
        Image.fromarray(foliage_mask.astype(np.uint8) * 255).save(
            run / "masks" / f"{Path(name).stem}.foliage.png"
        )

        if not args.skip_previews:
            painted = rgb.copy()
            fg = foliage_mask
            painted[fg, 1] = np.clip(painted[fg, 1].astype(np.float32) * 0.45 + 140, 0, 255).astype(
                np.uint8
            )
            review = Image.fromarray(painted)
            draw = ImageDraw.Draw(review)
            x0, y0, x1, y1 = (
                int(info["box_x0"]),
                int(info["box_y0"]),
                int(info["box_x1"]),
                int(info["box_y1"]),
            )
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(0, 220, 255), width=6)
            review.thumbnail((1600, 1600))
            review.save(overlay_dir / f"{Path(name).stem}_foliage.jpg", quality=85)

    cached_name = None
    foliage = None
    for row in tiles:
        path = tile_dir / row["tile"]
        tile_rgb = np.asarray(Image.open(path).convert("RGB"))
        info = images[row["image"]]
        x, y, width, height = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
        if row["image"] != cached_name:
            stem = Path(row["image"]).stem
            foliage = np.asarray(Image.open(mask_dir / f"{stem}.foliage.png")) > 127
            cached_name = row["image"]
        shape = tile_rgb.shape[:2]
        # Foliage pixels above the tube line: pale or bronzed flush counts here
        # even when it is not green enough for veg_frac. White plastic never does.
        foliage_patch = tile_patch(foliage, x, y, width, height, shape)
        upper_rows = max(0, min(height, tube_start_by_image[row["image"]] - y))
        upper_rows = int(round(upper_rows * shape[0] / max(1, height)))
        foliage_frac = float(foliage_patch.mean())
        upper_foliage_frac = float(foliage_patch[:upper_rows].sum() / foliage_patch.size)

        veg_frac = float(vegetation(tile_rgb).mean())
        white_frac = float(white_tube(tile_rgb).mean())
        rel_y = (y - int(info["box_y0"])) / max(1, int(info["box_h"]))
        is_tube_tile = (
            rel_y >= TUBE_Y_FRAC and white_frac >= TUBE_WHITE and veg_frac < TUBE_VEG
        )
        keep = (
            foliage_frac >= MIN_FOLIAGE_MASK
            and max(veg_frac, upper_foliage_frac) >= MIN_VEG_FRAC
            and not is_tube_tile
        )
        if keep:
            shutil.copy2(path, foliage_dir / row["tile"])
            kept_by_image[row["image"]].append(foliage_dir / row["tile"])
            decision = "keep"
        else:
            shutil.copy2(path, dropped_dir / row["tile"])
            decision = "drop"
        out_rows.append(
            {
                **row,
                "foliage_mask_frac": round(foliage_frac, 4),
                "veg_frac": round(veg_frac, 4),
                "upper_foliage_frac": round(upper_foliage_frac, 4),
                "white_frac": round(white_frac, 4),
                "rel_y": round(rel_y, 3),
                "decision": decision,
            }
        )

    with (run / "tiles_foliage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(out_rows)

    if not args.skip_previews:
        for name, paths in kept_by_image.items():
            contact_sheet(paths).save(sheet_dir / f"{Path(name).stem}_foliage.jpg", quality=85)

    kept = sum(1 for row in out_rows if row["decision"] == "keep")
    dropped = len(out_rows) - kept
    per_image = [len(paths) for paths in kept_by_image.values()] or [0]
    summary = {
        "input_tiles": len(out_rows),
        "foliage_kept": kept,
        "dropped": dropped,
        "per_image_min": int(min(per_image)),
        "per_image_max": int(max(per_image)),
        "per_image_mean": round(float(np.mean(per_image)), 2),
        "min_foliage_mask": MIN_FOLIAGE_MASK,
        "min_veg_frac": MIN_VEG_FRAC,
        "tube_y_frac": TUBE_Y_FRAC,
        "pale_upper_flush_kept": True,
    }
    (run / "foliage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
