"""Training package for visible thrips-damage severity."""

import os
from pathlib import Path

# Keep model downloads with the training artifacts on the server. This is set
# before any training submodule imports Transformers/Hugging Face.
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent / "model_cache"))

__version__ = "0.1.0"
