from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATA = ROOT / "data"
SETTINGS_PATH = DATA / "settings.json"


@dataclass
class Settings:
    annotator: str = "lab"
    pipeline_mode: str = "ssh"
    ssh_host: str = "10.248.22.167"
    ssh_user: str = "fpt"
    ssh_password: str = ""
    remote_project: str = "/home/fpt/RaghavWork/Segmentation_ResearchPaper"
    remote_python: str = "/home/fpt/RaghavWork/Segmentation_ResearchPaper/.venv/bin/python"
    remote_work: str = "/home/fpt/Chili thrips detection pictures/annotator_jobs"
    remote_existing_run: str = (
        "/home/fpt/Chili thrips detection pictures/analysis_2026-08-17/birefnet_tiles"
    )
    local_existing_run: str = ""
    local_project: str = ""
    local_python: str = "python3"
    device: str = "cuda:0"
    fold: int = 0


def _apply_env(settings: Settings) -> Settings:
    mapping = {
        "ANNOTATOR_NAME": "annotator",
        "ANNOTATOR_PIPELINE_MODE": "pipeline_mode",
        "ANNOTATOR_SSH_HOST": "ssh_host",
        "ANNOTATOR_SSH_USER": "ssh_user",
        "ANNOTATOR_SSH_PASSWORD": "ssh_password",
        "ANNOTATOR_REMOTE_PROJECT": "remote_project",
        "ANNOTATOR_REMOTE_PYTHON": "remote_python",
        "ANNOTATOR_REMOTE_WORK": "remote_work",
        "ANNOTATOR_REMOTE_EXISTING_RUN": "remote_existing_run",
        "ANNOTATOR_LOCAL_EXISTING_RUN": "local_existing_run",
        "ANNOTATOR_LOCAL_PROJECT": "local_project",
        "ANNOTATOR_LOCAL_PYTHON": "local_python",
        "ANNOTATOR_DEVICE": "device",
    }
    for env_name, field_name in mapping.items():
        value = os.environ.get(env_name)
        if value:
            setattr(settings, field_name, value)
    fold = os.environ.get("ANNOTATOR_FOLD")
    if fold:
        settings.fold = int(fold)
    return settings


def load_settings() -> Settings:
    settings = Settings()
    if SETTINGS_PATH.exists():
        raw = json.loads(SETTINGS_PATH.read_text())
        known = {item.name for item in fields(Settings)}
        for key, value in raw.items():
            if key in known:
                setattr(settings, key, value)
    return _apply_env(settings)


def save_settings(settings: Settings) -> Settings:
    DATA.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(asdict(settings), indent=2) + "\n")
    return settings


def public_settings(settings: Settings) -> dict:
    payload = asdict(settings)
    payload["ssh_password_set"] = bool(settings.ssh_password)
    return payload
