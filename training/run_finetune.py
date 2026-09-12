from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import json_dump


FOLDS = {0: [0, 2, 4], 1: [1, 3]}
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=sorted(FOLDS))
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path, default=Path("training/artifacts/splits.csv"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    artifacts = root / "training" / "artifacts"
    slug = MODEL_ID.replace("/", "__")
    status_path = artifacts / f"finetune_gpu{args.gpu}_status.json"
    status = {"gpu": args.gpu, "model_id": MODEL_ID, "folds": FOLDS[args.gpu],
              "state": "running", "started_at": now(), "finished_at": None,
              "current": None, "completed": [], "failed": []}
    json_dump(status, status_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    for fold in FOLDS[args.gpu]:
        output = artifacts / slug / f"fold_{fold}" / "finetune"
        checkpoint = output / "best.pt"
        if checkpoint.is_file() and (output / "history.json").is_file():
            status["completed"].append({"fold": fold, "state": "already_complete"})
            json_dump(status, status_path); continue
        probe = artifacts / slug / f"fold_{fold}" / "probe" / "best.pt"
        log = artifacts / "logs" / f"gpu{args.gpu}_{slug}_fold{fold}_finetune.log"
        command = [
            sys.executable, "-m", "training.train_tiles", "--dataset-root", str(root),
            "--splits", str(args.splits.resolve()), "--fold", str(fold),
            "--stage", "finetune", "--model-id", MODEL_ID,
            "--init-checkpoint", str(probe), "--workers", str(args.workers),
            "--output-root", str(artifacts),
        ]
        status["current"] = {"fold": fold, "started_at": now(), "log": str(log)}
        json_dump(status, status_path)
        with log.open("a") as handle:
            handle.write(f"START {now()} {json.dumps(command)}\n"); handle.flush()
            result = subprocess.run(command, cwd=root, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT, check=False)
            handle.write(f"END {now()} exit_code={result.returncode}\n")
        record = {"fold": fold, "exit_code": result.returncode, "finished_at": now(), "log": str(log)}
        if result.returncode:
            status["failed"].append(record); status["state"] = "failed"
            status["finished_at"] = now(); status["current"] = None
            json_dump(status, status_path); raise SystemExit(result.returncode)
        status["completed"].append(record); status["current"] = None
        json_dump(status, status_path)
    status["state"] = "complete"; status["finished_at"] = now(); status["current"] = None
    json_dump(status, status_path)


if __name__ == "__main__":
    main()
