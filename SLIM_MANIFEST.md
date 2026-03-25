# Xsim_min Slim Bundle

This bundle keeps only the current `spherical_proxy` training path and the minimum
files needed to run it.

Included:
- `train.py`
- `run_joint_proxy_train.sh`
- `configs/hdl64e_gsplat_joint_proxy.yaml`
- `configs/pandaset.yaml`
- `scripts/lidar_smoke_test.py`
- `src/three_dgut_gsplat_min/`
- `submodules/gsplat_upstream_clean/gsplat/`
- `submodules/gsplat_upstream_clean/{setup.py,pyproject.toml,MANIFEST.in,LICENSE,README.md}`

Excluded on purpose:
- any `checkpoints*` directories
- any `logs*` directories
- debug / analysis scripts not needed for the current render path
- gsplat docs, examples, tests, and other non-runtime bulk removed after copy

Current active LiDAR render backend in this bundle:
- `spherical_proxy`

Dataset roots are no longer hardcoded in the YAML configs. Pass them at runtime with:

```bash
python3 train.py --config configs/pandaset.yaml --dataset-root /path/to/pandaset
```

or with the launcher:

```bash
bash run_joint_proxy_train.sh /path/to/kitti-root
```

Minimal smoke command validated in this repo:

```bash
PYTHONPATH=src:submodules/gsplat_upstream_clean python3 - <<'PY'
from three_dgut_gsplat_min import JointTrainer, load_config
cfg = load_config('configs/pandaset.yaml', dataset_root='datasets/pandaset/pandaset')
cfg.training.max_steps = 1
cfg.training.checkpoint_dir = 'checkpoints_pandaset_smoke_cli'
cfg.training.vis_dir = 'checkpoints_pandaset_smoke_cli/vis'
trainer = JointTrainer(cfg)
trainer.train()
PY
```
