from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
GSPLAT_ROOT = Path(os.environ.get("GSPLAT_ROOT", "/mnt/data16/xuzhiy/gsplat_upstream_clean")).expanduser()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSPLAT_ROOT))

from three_dgut_gsplat_min import JointTrainer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3DGUT-style joint LiDAR/RGB gsplat model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Override dataset.source_path from the YAML config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, dataset_root=args.dataset_root)
    trainer = JointTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
