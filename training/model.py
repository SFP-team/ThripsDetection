from __future__ import annotations

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


def pooled_embedding(outputs: object) -> torch.Tensor:
    pooler = getattr(outputs, "pooler_output", None)
    if pooler is not None:
        return pooler
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        raise ValueError("backbone returned neither pooler_output nor last_hidden_state")
    if hidden.ndim == 4:  # ConvNeXt: B,C,H,W
        return hidden.mean(dim=(-2, -1))
    if hidden.ndim == 3:  # ViT: CLS token
        return hidden[:, 0]
    if hidden.ndim == 2:
        return hidden
    raise ValueError(f"unsupported backbone output shape: {tuple(hidden.shape)}")


def hidden_size(config: object) -> int:
    value = getattr(config, "hidden_size", None)
    if value:
        return int(value)
    values = getattr(config, "hidden_sizes", None)
    if values:
        return int(values[-1])
    raise ValueError("cannot infer backbone feature dimension")


class ThripsTileModel(nn.Module):
    def __init__(self, model_id: str, dropout: float = 0.2, local_files_only: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        self.backbone = AutoModel.from_pretrained(model_id, config=config, local_files_only=local_files_only)
        width = hidden_size(config)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.tissue_head = nn.Linear(width, 1)
        self.damage_head = nn.Linear(width, 2)  # P(y>healthy), P(y>mild)
        self.model_id = model_id

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.norm(pooled_embedding(self.backbone(pixel_values=pixel_values))))
        return {
            "tissue_logit": self.tissue_head(features).squeeze(-1),
            "damage_logits": self.damage_head(features),
        }

    def freeze_backbone(self) -> None:
        self.backbone.requires_grad_(False)

    def unfreeze_last_stage(self) -> list[str]:
        self.backbone.requires_grad_(False)
        candidates = [
            "model.stages",
            "encoder.stages",
            "stages",
            "model.layer",
            "encoder.layer",
            "layer",
        ]
        selected: list[str] = []
        for dotted in candidates:
            module: object = self.backbone
            try:
                for part in dotted.split("."):
                    module = getattr(module, part)
                blocks = list(module)  # type: ignore[arg-type]
            except (AttributeError, TypeError):
                continue
            take = 2 if "layer" in dotted and "stages" not in dotted else 1
            for block in blocks[-take:]:
                block.requires_grad_(True)
            selected.append(f"{dotted}[-{take}:]")
            break
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.LayerNorm) and ("norm" in name or "layernorm" in name):
                module.requires_grad_(True)
        if not selected:
            raise RuntimeError("could not locate the final backbone stage to unfreeze")
        return selected


def coral_targets(labels: torch.Tensor) -> torch.Tensor:
    """Convert class 0/1/2 to cumulative ordinal targets."""
    return torch.stack((labels > 0, labels > 1), dim=1).float()


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    cumulative = torch.sigmoid(logits)
    q1 = cumulative[:, 0]
    q2 = torch.minimum(q1, cumulative[:, 1])
    probs = torch.stack((1.0 - q1, q1 - q2, q2), dim=1)
    return probs.clamp_min(0.0)
