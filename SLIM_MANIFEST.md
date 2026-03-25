# Xsim_min Slim Bundle

This bundle keeps only the current `spherical_proxy` training path and the minimum
files needed to run it.

Included:
- `train.py`
- `run_joint_proxy_train.sh`
- `configs/hdl64e_gsplat_joint_proxy.yaml`
- `scripts/lidar_smoke_test.py`
- `src/three_dgut_gsplat_min/`
- `third_party/gsplat_upstream_clean/gsplat/`
- `third_party/gsplat_upstream_clean/{setup.py,pyproject.toml,MANIFEST.in,LICENSE,README.md}`

Excluded on purpose:
- any `checkpoints*` directories
- any `logs*` directories
- debug / analysis scripts not needed for the current render path
- gsplat docs, examples, tests, and other non-runtime bulk removed after copy

Current active LiDAR render backend in this bundle:
- `spherical_proxy`
