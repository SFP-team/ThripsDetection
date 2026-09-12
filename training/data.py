from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor


DAMAGE_TO_ID = {"healthy": 0, "mild": 1, "severe": 2}


def build_transform(model_id: str, size: int, train: bool):
    processor = AutoImageProcessor.from_pretrained(model_id)
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    operations: list[object] = [transforms.Resize((size, size), InterpolationMode.BICUBIC)]
    if train:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=8, translate=(0.05, 0.05), scale=(0.9, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08, hue=0.02),
                transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.1),
            ]
        )
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return transforms.Compose(operations)


class TileDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, root: Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows.iloc[index]
        with Image.open(self.root / row.tile_path) as handle:
            pixel_values = self.transform(handle.convert("RGB"))
        # Tissue and injury are present during training/evaluation, but a new
        # field run has only image metadata. Dummy targets keep one dataset
        # implementation usable for both cases; inference ignores them.
        tissue = str(row.get("tissue", ""))
        injury = str(row.get("injury", ""))
        return {
            "pixel_values": pixel_values,
            "tissue_target": torch.tensor(float(tissue == "flush")),
            "damage_target": torch.tensor(DAMAGE_TO_ID.get(injury, -1), dtype=torch.long),
            "tile": str(row.tile),
            "image": str(row.image),
        }


def plant_class_sampler(rows: pd.DataFrame, seed: int) -> WeightedRandomSampler:
    plant_counts = Counter(rows["image"])
    strata = rows.apply(
        lambda r: r.injury if r.tissue == "flush" and r.injury in DAMAGE_TO_ID else f"tissue_{r.tissue}", axis=1
    )
    class_counts = Counter(strata)
    median_count = float(pd.Series(list(class_counts.values())).median())
    weights = []
    for image, stratum in zip(rows["image"], strata):
        class_factor = min(4.0, (median_count / class_counts[stratum]) ** 0.5)
        weights.append(class_factor / plant_counts[image])
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(rows), replacement=True, generator=generator
    )
