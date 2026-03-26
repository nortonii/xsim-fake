# XSIM Min Slim

This repository contains the slimmed LiDAR/RGB joint training pipeline used for KITTI, PandaSet, and Waymo experiments.

## Quick Start

The main entry point is `run_joint_proxy_train.sh`.

It accepts the dataset root as the first positional argument:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHON_BIN=/mnt/data16/xuzhiy/HiGS-Calib/HiGS/bin/python \
bash run_joint_proxy_train.sh /path/to/dataset_root
```

You can also override the config file explicitly:

```bash
CUDA_VISIBLE_DEVICES=3 \
PYTHON_BIN=/mnt/data16/xuzhiy/HiGS-Calib/HiGS/bin/python \
CONFIG_PATH=configs/hdl64e_gsplat_joint_proxy_kitti_interp_smoothl1.yaml \
bash run_joint_proxy_train.sh /path/to/kitti-calibration
```

If you prefer environment variables instead of a positional argument:

```bash
export DATASET_ROOT=/path/to/dataset_root
export CONFIG_PATH=configs/hdl64e_gsplat_joint_proxy_kitti_interp_smoothl1.yaml
export PYTHON_BIN=/mnt/data16/xuzhiy/HiGS-Calib/HiGS/bin/python
bash run_joint_proxy_train.sh
```

## Current Formal Configs

- KITTI: `configs/hdl64e_gsplat_joint_proxy_kitti_interp_smoothl1.yaml`
- PandaSet: `configs/pandaset.yaml`
- Waymo: `configs/waymo.yaml`

The KITTI launcher defaults to the smooth-L1 configuration above.

## Notes

- `run_joint_proxy_train.sh` forwards the dataset root to `train.py --dataset-root`.
- `TORCH_EXTENSIONS_DIR` defaults to `/tmp/torch_extensions`.
- Temporary smoke-test YAML files are not kept in the repository; use the formal configs above for repeatable runs.
