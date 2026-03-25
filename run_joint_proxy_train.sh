#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
export PYTHONPATH="${PYTHONPATH:-src:submodules/gsplat_upstream_clean}"
CONFIG_PATH="${CONFIG_PATH:-configs/hdl64e_gsplat_joint_proxy_kitti_interp.yaml}"
DATASET_ROOT="${DATASET_ROOT:-}"
if [[ $# -gt 0 ]]; then
  DATASET_ROOT="$1"
  shift
fi
if [[ -z "${DATASET_ROOT}" ]]; then
  echo "Usage: $0 <dataset_root> [extra train.py args...]" >&2
  echo "Or set DATASET_ROOT=/path/to/dataset before running." >&2
  exit 1
fi
exec "${PYTHON_BIN}" train.py --config "${CONFIG_PATH}" --dataset-root "${DATASET_ROOT}" "$@"
