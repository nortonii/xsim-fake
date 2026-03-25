from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
GSPLAT_ROOT = PROJECT_ROOT / "third_party" / "gsplat_upstream_clean"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSPLAT_ROOT))

from three_dgut_gsplat_min import JointTrainer, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3DGUT-style joint LiDAR/RGB gsplat model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    trainer = JointTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
