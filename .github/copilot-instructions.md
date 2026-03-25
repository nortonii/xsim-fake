# Copilot Instructions for `Xsim_min_slim`

## Commands

### Main training entrypoints

- Preferred launcher for the current default setup:
  ```bash
  bash run_joint_proxy_train.sh
  ```
  This runs `train.py` with `configs/hdl64e_gsplat_joint_proxy_kitti_interp.yaml` and sets:
  ```bash
  TORCH_EXTENSIONS_DIR=/tmp/torch_extensions
  PYTHONPATH=src:submodules/gsplat_upstream_clean
  ```

- Direct training command:
  ```bash
  python train.py --config configs/hdl64e_gsplat_joint_proxy_kitti_interp.yaml
  ```

- Train with a different config:
  ```bash
  python train.py --config configs/waymo.yaml
  ```

### Lightweight validation

- LiDAR render smoke test:
  ```bash
  python scripts/lidar_smoke_test.py --config configs/hdl64e_gsplat_joint_proxy_kitti_interp.yaml
  ```

- Smallest practical single-check command in this bundle:
  ```bash
  python scripts/lidar_smoke_test.py --config configs/waymo.yaml
  ```

### Packaging / dependency context

- The local Python package metadata is in `submodules/gsplat_upstream_clean/pyproject.toml`.
- Vendored `gsplat` also has its own `setup.py`; importing it may trigger CUDA extension builds/rebuilds.
- `pandaset` is expected to be installed outside the repo (for example: `pip install /path/to/pandaset-devkit/python`). Do not vendor or submodule PandaSet devkit inside this repository.

## High-level architecture

This repository is a slim training bundle focused on the current `spherical_proxy` / `spherical_proxy_ut` LiDAR path. The active local code is under `src/three_dgut_gsplat_min`; `submodules/gsplat_upstream_clean` is vendored upstream rendering infrastructure.

The top-level runtime flow is:

1. `train.py` loads a YAML config with `load_config()` and constructs `JointTrainer`.
2. `JointTrainer` selects one dataset implementation based on `dataset.mode`:
   - `manifest` -> `MultiSensorDataset`
   - `kitti_r` -> `KittiRDataset`
   - `waymo` -> `WaymoDataset`
3. The trainer builds `GaussianSceneModel`, configures the LiDAR renderer from dataset config, optionally initializes Gaussian means from aggregated LiDAR points, then runs joint RGB + LiDAR optimization.
4. `JointLoss` combines RGB reconstruction (`L1` plus optional SSIM) with LiDAR depth supervision and regularizers.
5. Training periodically writes checkpoints plus visualization and geometry exports under each config’s `training.checkpoint_dir` / `training.vis_dir`.

### Key modules

- `src/three_dgut_gsplat_min/config.py`
  Central dataclass-based config schema. Most behavior is driven from YAML.

- `src/three_dgut_gsplat_min/trainer.py`
  Orchestrates dataset loading, deterministic train/test split, optimizer setup, optional `gsplat` densification strategy, training loop, checkpointing, visualizations, and geometry export.

- `src/three_dgut_gsplat_min/model.py`
  Defines `GaussianSceneModel`. RGB rendering uses `gsplat` rasterization; LiDAR rendering uses local spherical projection code paths plus optional UT-based footprint logic.

- `src/three_dgut_gsplat_min/data.py` and `data_waymo.py`
  Normalize different dataset layouts into a common `FrameSample` contract with camera pose, LiDAR pose, intrinsics, RGB, LiDAR depth, and optional raw LiDAR points / dynamic masks.

- `src/three_dgut_gsplat_min/losses.py`
  Implements the joint loss, including dynamic-mask-aware RGB loss and angle-table-weighted LiDAR sampling.

## Repository-specific conventions

### Dataset and pose semantics

- All dataset loaders are expected to return the same `FrameSample` semantics. Trainer/model/loss code assumes those field names, shapes, and pose meanings are stable.
- `KittiRDataset` normalizes LiDAR poses relative to the first frame and reconstructs `camera_to_world` from the LiDAR pose plus calibration transforms.
- `WaymoDataset` uses ego pose as the LiDAR pose proxy and optionally loads `dynamic_mask/*.png`; those masks are consumed by `JointLoss` to ignore dynamic RGB regions.

### LiDAR angle-table behavior is shared state

- `JointTrainer._ensure_fixed_vertical_angles()` may synthesize or upsample the LiDAR vertical angle table and writes it back into `config.dataset`.
- For fitted angles, trainer also mirrors dataset-estimated angles back into config so rendering, loss weighting, eval, and visualization all stay aligned.
- If you change LiDAR height, sensor type, angle mode, or interpolation factor, make sure the dataset, renderer, and loss still agree on the same row semantics.

### Default configs are tuned around the local spherical proxy path

- The slim bundle’s active path is documented in `SLIM_MANIFEST.md` as `spherical_proxy`.
- Current configs mostly use:
  - `dataset.mode: kitti_r` or `waymo`
  - `lidar_render_backend: spherical_proxy_ut`
  - `model.densify.strategy: mcmc`
  - `model.use_separate_opacity: false`
- `spherical_proxy_ut` currently only supports `lidar_depth_aggregation: mean`.

### Densification mutates model parameters

- `trainer.py` is intentionally careful about syncing `self.params`, `self.model`, and optimizer state because `gsplat` densification strategies can replace parameter tensors.
- If you modify optimizer wiring, opacity handling, or densification logic, preserve the existing parameter-sync behavior or training/checkpointing will silently diverge.

### Initialization and outputs matter

- For `kitti_r` and `waymo`, the model usually initializes Gaussian means from aggregated LiDAR points rather than random scene extents.
- Checkpoint directories also contain visualization artifacts and geometry exports:
  - `step_*.pt` checkpoints
  - `vis/step_*/` PNG/NPY debug outputs
  - `geometry/` PLY + pose exports
- Several code paths treat those outputs as part of the normal workflow, not as disposable debug-only files.

### Vendored upstream code is not the main editing target

- Prefer making changes in `src/three_dgut_gsplat_min/` and configs unless the task explicitly requires patching vendored `gsplat`.
- This bundle removed most upstream docs/tests/examples, so do not assume upstream `gsplat` development commands or test paths exist locally.
