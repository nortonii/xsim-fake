from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
GSPLAT_ROOT = PROJECT_ROOT / "third_party" / "gsplat_upstream_clean"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(GSPLAT_ROOT) not in sys.path:
    sys.path.insert(0, str(GSPLAT_ROOT))

from three_dgut_gsplat_min.config import load_config
from three_dgut_gsplat_min.data import KittiRDataset
from three_dgut_gsplat_min.model import GaussianSceneModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "hdl64e_gsplat_min.yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KittiRDataset(
        source_path=cfg.dataset.source_path,
        data_seq=cfg.dataset.data_seq,
        cam_id=cfg.dataset.cam_id,
        start_index=cfg.dataset.start_index,
        data_type=cfg.dataset.data_type,
        segment_length=cfg.dataset.segment_length,
        lidar_width=cfg.dataset.lidar_width,
        lidar_height=cfg.dataset.lidar_height,
        max_range=cfg.dataset.max_range,
        near_plane=0.0,
        far_plane=1.0e9,
        lidar_vertical_fov_min_deg=cfg.dataset.lidar_vertical_fov_min_deg,
        lidar_vertical_fov_max_deg=cfg.dataset.lidar_vertical_fov_max_deg,
        lidar_vertical_angles_deg=cfg.dataset.lidar_vertical_angles_deg,
        lidar_angle_mode=cfg.dataset.lidar_angle_mode,
        lidar_dbxml_path=None,
    )
    if getattr(dataset, "lidar_vertical_angles_deg", None):
        cfg.dataset.lidar_vertical_angles_deg = list(dataset.lidar_vertical_angles_deg)

    sample = dataset[0]
    gt_depth = sample.lidar_depth.squeeze(0)
    gt_rows = (gt_depth > 0).nonzero(as_tuple=False)[:, 0]

    model = GaussianSceneModel(
        num_gaussians=cfg.model.num_gaussians,
        sh_degree=cfg.model.sh_degree,
        init_extent=cfg.model.init_extent,
        init_opacity=cfg.model.init_opacity,
        init_scale=cfg.model.init_scale,
        background_color=cfg.model.background_color,
        use_separate_opacity=cfg.model.use_separate_opacity,
        lidar_ut_enable=cfg.model.lidar_ut_enable,
        lidar_ut_alpha=cfg.model.lidar_ut_alpha,
        lidar_ut_beta=cfg.model.lidar_ut_beta,
        lidar_ut_kappa=cfg.model.lidar_ut_kappa,
        lidar_ut_delta=cfg.model.lidar_ut_delta,
        lidar_ut_in_image_margin_factor=cfg.model.lidar_ut_in_image_margin_factor,
        lidar_ut_require_all_sigma_points_valid=cfg.model.lidar_ut_require_all_sigma_points_valid,
    ).to(device)
    model.configure_lidar_model(cfg.dataset)

    with torch.no_grad():
        if model.use_separate_opacity:
            model.opacity_camera.data.fill_(20.0)
            model.opacity_lidar.data.fill_(20.0)
        else:
            model.opacities.data.fill_(20.0)

    render = model.render_lidar(
        lidar_to_world=sample.lidar_to_world.to(device),
        width=cfg.dataset.lidar_width,
        height=cfg.dataset.lidar_height,
        vertical_fov_min_deg=cfg.dataset.lidar_vertical_fov_min_deg,
        vertical_fov_max_deg=cfg.dataset.lidar_vertical_fov_max_deg,
        near_plane=0.0,
        far_plane=1.0e9,
        vertical_angles_deg=cfg.dataset.lidar_vertical_angles_deg,
        lidar_vertical_angles_deg=cfg.dataset.lidar_vertical_angles_deg,
    )

    depth = render.depth.detach().cpu()
    alpha = render.alpha.detach().cpu()
    render_rows = (depth.squeeze(0) > 0).nonzero(as_tuple=False)[:, 0]

    print(f"config: {args.config}")
    print(f"gt shape: {tuple(gt_depth.shape)} rows={int(gt_rows.min())}..{int(gt_rows.max())}")
    print(f"gt top10: {Counter(gt_rows.tolist()).most_common(10)}")
    print(f"render shape: {tuple(depth.shape)} alpha_shape={tuple(alpha.shape)}")
    print(f"render depth: {float(depth.min()):.4f}..{float(depth.max()):.4f}")
    print(f"render alpha: {float(alpha.min()):.4f}..{float(alpha.max()):.4f}")
    print(f"render rows={int(render_rows.min())}..{int(render_rows.max())}")
    print(f"render top10: {Counter(render_rows.tolist()).most_common(10)}")


if __name__ == "__main__":
    main()
