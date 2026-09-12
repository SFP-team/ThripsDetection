from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import json_dump, load_config


QUEUES = {
    0: [
        ("facebook/dinov3-convnext-tiny-pretrain-lvd1689m", [0, 1, 2, 3, 4]),
        ("facebook/dinov2-with-registers-small", [0, 1, 2]),
    ],
    1: [
        ("facebook/dinov3-vits16-pretrain-lvd1689m", [0, 1, 2, 3, 4]),
        ("facebook/dinov2-with-registers-small", [3, 4]),
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=sorted(QUEUES))
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path, default=Path("training/artifacts/splits.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    artifacts = root / "training" / "artifacts"
    logs = artifacts / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    status_path = artifacts / f"bakeoff_gpu{args.gpu}_status.json"
    status = {
        "gpu": args.gpu,
        "started_at": utc_now(),
        "finished_at": None,
        "state": "running",
        "current": None,
        "completed": [],
        "failed": [],
        "queue": [{"model_id": model, "folds": folds} for model, folds in QUEUES[args.gpu]],
    }
    json_dump(status, status_path)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    for model_id, folds in QUEUES[args.gpu]:
        model_slug = model_id.replace("/", "__")
        for fold in folds:
            checkpoint = artifacts / model_slug / f"fold_{fold}" / "probe" / "best.pt"
            history = checkpoint.with_name("history.json")
            if checkpoint.is_file() and history.is_file():
                status["completed"].append(
                    {"model_id": model_id, "fold": fold, "state": "already_complete"}
                )
                json_dump(status, status_path)
                continue
            log_path = logs / f"gpu{args.gpu}_{model_slug}_fold{fold}_probe.log"
            status["current"] = {
                "model_id": model_id, "fold": fold, "started_at": utc_now(), "log": str(log_path)
            }
            json_dump(status, status_path)
            command = [
                sys.executable, "-m", "training.train_tiles",
                "--dataset-root", str(root),
                "--splits", str(args.splits.resolve()),
                "--fold", str(fold),
                "--stage", "probe",
                "--model-id", model_id,
                "--workers", str(args.workers),
                "--output-root", str(artifacts),
            ]
            with log_path.open("a") as handle:
                handle.write(f"\nSTART {utc_now()} command={json.dumps(command)}\n")
                handle.flush()
                result = subprocess.run(
                    command, cwd=root, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                    check=False,
                )
                handle.write(f"END {utc_now()} exit_code={result.returncode}\n")
            record = {
                "model_id": model_id, "fold": fold, "finished_at": utc_now(),
                "exit_code": result.returncode, "log": str(log_path),
            }
            if result.returncode:
                status["failed"].append(record)
                status["state"] = "failed"
                status["current"] = None
                status["finished_at"] = utc_now()
                json_dump(status, status_path)
                raise SystemExit(result.returncode)
            status["completed"].append(record)
            status["current"] = None
            json_dump(status, status_path)

    status["state"] = "complete"
    status["current"] = None
    status["finished_at"] = utc_now()
    json_dump(status, status_path)
    print(f"GPU {args.gpu} bake-off queue complete")


if __name__ == "__main__":
    main()
