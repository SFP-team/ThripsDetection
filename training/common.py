from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    return json.loads((PACKAGE_ROOT / "config.json").read_text())


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def image_stem(value: str) -> str:
    return Path(str(value)).stem.upper()


def json_dump(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def require_columns(columns: Iterable[str], required: set[str], source: str) -> None:
    missing = required - set(columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")

