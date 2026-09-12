from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .common import json_dump


QUEUES = {
    0: [
        "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
        "facebook/dinov2-with-registers-small",
    ],
    1: ["facebook/dinov3-vits16-pretrain-lvd1689m"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=sorted(QUEUES))
    parser.add_argument("--dataset-root", type=Path, default=Path("."))
    parser.add_argument("--splits", type=Path, default=Path("training/artifacts/splits.csv"))
    parser.add_argument("--stage", choices=["probe", "finetune"], default="probe")
    parser.add_argument("--models", nargs="*")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    artifacts = root / "training" / "artifacts"
    models = args.models or QUEUES[args.gpu]
    status_path = artifacts / f"oof_{args.stage}_gpu{args.gpu}_status.json"
    status = {"gpu": args.gpu, "stage": args.stage, "state": "running", "started_at": now(),
              "finished_at": None, "current": None, "completed": [], "failed": []}
    json_dump(status, status_path)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for model_id in models:
        slug = model_id.replace("/", "__")
        output = artifacts / slug / f"oof_{args.stage}_tiles.csv"
        for fold in range(5):
            checkpoint = artifacts / slug / f"fold_{fold}" / args.stage / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            status["current"] = {"model_id": model_id, "fold": fold, "started_at": now()}
            json_dump(status, status_path)
            command = [
                sys.executable, "-m", "training.predict_tiles", "--dataset-root", str(root),
                "--splits", str(args.splits.resolve()), "--fold", str(fold),
                "--stage", args.stage, "--checkpoint", str(checkpoint), "--output", str(output),
            ]
            log = artifacts / "logs" / f"gpu{args.gpu}_{slug}_fold{fold}_{args.stage}_predict.log"
            with log.open("a") as handle:
                handle.write(f"START {now()} {json.dumps(command)}\n"); handle.flush()
                result = subprocess.run(command, cwd=root, env=env, stdout=handle,
                                        stderr=subprocess.STDOUT, check=False)
                handle.write(f"END {now()} exit_code={result.returncode}\n")
            record = {"model_id": model_id, "fold": fold, "exit_code": result.returncode,
                      "finished_at": now(), "log": str(log)}
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

