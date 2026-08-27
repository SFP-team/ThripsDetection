#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/fpt/ThripsDetection}"
BASE_PYTHON="${TRAINING_BASE_PYTHON:-/home/fpt/RaghavWork/Segmentation_ResearchPaper/.venv/bin/python}"
ENV_ROOT="${PROJECT_ROOT}/.training-venv"

if [[ ! -x "${BASE_PYTHON}" ]]; then
  echo "Base CUDA Python not found: ${BASE_PYTHON}" >&2
  exit 2
fi

if [[ ! -x "${ENV_ROOT}/bin/python" ]]; then
  python3 -m venv "${ENV_ROOT}"
fi

BASE_SITE="$(${BASE_PYTHON} -c 'import site; print(site.getsitepackages()[0])')"
ENV_SITE="$(${ENV_ROOT}/bin/python -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n' "${BASE_SITE}" > "${ENV_SITE}/thrips_cuda_base.pth"

"${ENV_ROOT}/bin/python" - <<'PY'
import torch
import torchvision
import transformers
import pandas
import sklearn

assert torch.cuda.is_available(), "CUDA is not visible"
print(
    "READY",
    "torch", torch.__version__,
    "torchvision", torchvision.__version__,
    "transformers", transformers.__version__,
    "pandas", pandas.__version__,
    "sklearn", sklearn.__version__,
    "gpus", torch.cuda.device_count(),
)
PY

