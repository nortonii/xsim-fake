#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}"
export PYTHONPATH="${PYTHONPATH:-src:submodules/gsplat_upstream_clean}"
exec "${PYTHON_BIN}" train.py --config configs/hdl64e_gsplat_joint_proxy_kitti_interp.yaml
