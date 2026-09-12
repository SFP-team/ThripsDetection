from __future__ import annotations

import argparse

import torch
from transformers import AutoImageProcessor, AutoModel

from .common import load_config


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=cfg["primary_model_id"])
    parser.add_argument("--fallback-on-auth-error", action="store_true")
    args = parser.parse_args()
    model_id = args.model_id
    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    except Exception as exc:
        if not args.fallback_on_auth_error or model_id == cfg["fallback_model_id"]:
            raise
        print(f"Primary model unavailable ({type(exc).__name__}); downloading public fallback.")
        model_id = cfg["fallback_model_id"]
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
    size = int(cfg["input_size"])
    with torch.inference_mode():
        output = model(pixel_values=torch.zeros(1, 3, size, size))
    shape = tuple(output.last_hidden_state.shape)
    print(f"READY model_id={model_id} input={size} output={shape} processor={type(processor).__name__}")


if __name__ == "__main__":
    main()

