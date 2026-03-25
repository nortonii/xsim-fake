from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    # Dataset selection
    mode: str = "manifest"  # "manifest" (default) or "kitti_r" or "waymo" or "pandaset"

    # Manifest-mode parameters
    manifest_path: str | None = None
    rgb_tensor_key: str | None = None
    lidar_depth_key: str | None = None

    # KITTI_R-mode parameters (HiGS-Calib compatible layout)
    source_path: str | None = None
    data_seq: str | None = None
    cam_id: str = "02"
    start_index: int = 0
    data_type: str = "t"
    segment_length: int = 50
    lidar_width: int = 2048
    lidar_height: int = 64
    max_range: float = 50.0

    # Common sensor parameters
    near_plane: float = 0.1
    far_plane: float = 120.0
    lidar_vertical_fov_min_deg: float = -15.0
    lidar_vertical_fov_max_deg: float = 15.0
    # Optional per-ring vertical angles (degrees), length should match lidar_height.
    lidar_vertical_angles_deg: list[float] | None = None
    # Global vertical angle bias applied after loading/fitting the angle table.
    lidar_vertical_angle_offset_deg: float = 0.0
    # Sensor type for built-in angle tables: "hdl64e", "waymo", "waymo_top", "pandar128"
    lidar_sensor: str = "hdl64e"
    # Angle mode: "fixed" uses lidar_vertical_angles_deg; "fitted" uses data-estimated angles.
    lidar_angle_mode: str = "fixed"
    # Optional interpolation factor for vertical angles.
    # When > 1, the current angle table is densified by linear interpolation.
    lidar_vertical_angles_interp_factor: int = 1
    # Optional db.xml calibration file for fitted angle estimation.
    lidar_dbxml_path: str | None = None
    # LiDAR render backend:
    # - "custom": existing spherical splat path
    # - "gsplat_ut": native gsplat lidar camera model + UT projection
    # - "gsplat_ut_proj" / "spherical_proxy": legacy differentiable spherical projection
    #   + generic 2D rasterization proxy
    # - "spherical_proxy_ut": local UT-style spherical proxy path without gsplat UT
    lidar_render_backend: str = "custom"
    # Spherical LiDAR splat mode: "bilinear" (4-neighbor) or "gaussian" (uses radius).
    lidar_spherical_splat_mode: str = "bilinear"
    # Spherical splat kernel radius in pixels (used when mode="gaussian").
    lidar_spherical_kernel_radius: int = 1
    lidar_row_elevations_deg: list[float] | None = None
    lidar_spinning_frequency_hz: float = 10.0
    lidar_spinning_direction: str = "clockwise"  # "clockwise" or "counterclockwise"
    lidar_column_azimuth_start_deg: float = 180.0
    lidar_column_azimuth_end_deg: float = -180.0
    # LiDAR depth aggregation mode: "mean" uses the existing spherical splat path,
    # "median" reuses gsplat's 2DGS median-depth backend on the same synthetic projection.
    lidar_depth_aggregation: str = "mean"
    # Median-depth transmittance threshold passed to gsplat's 2DGS median backend.
    lidar_median_threshold: float = 0.5
    # Axis transform before gsplat_lidar projection.
    # Options: "xyz", "x(-y)z", "xy(-z)", "x(-y)(-z)", "yxz"
    lidar_axis_mode: str = "xyz"

    # Waymo-mode parameters (source_path should point to ".../waymo")
    waymo_scene_id: str | None = None

    # PandaSet-mode parameters
    pandaset_sequence_id: str | None = None
    pandaset_camera_name: str = "front_camera"
    pandaset_lidar_sensor_id: int = 0


@dataclass
class DensifyConfig:
    enable: bool = True
    strategy: str = "mcmc"  # "default" or "mcmc"

    # DefaultStrategy hyperparams (gsplat)
    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: str = "means2d"  # "means2d" or "gradient_2dgs"

    # State init
    scene_scale: float | None = None  # if None, use dataset.max_range


@dataclass
class ModelConfig:
    num_gaussians: int = 50000
    sh_degree: int = 3
    init_extent: float = 8.0
    init_opacity: float = 0.1
    init_scale: float = 0.02
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # NOTE: must be false when using gsplat densification strategies.
    use_separate_opacity: bool = False

    # Initialization options (useful for KITTI_R where poses are far from origin)
    init_from_lidar: bool = True
    init_lidar_frames: int = 20
    save_init_ply: bool = False

    # Unscented projection (3DGUT-style) for LiDAR splat.
    # This is only consumed by the native gsplat UT backend; the local
    # spherical_proxy_ut backend has its own projection code path.
    lidar_ut_enable: bool = False
    lidar_ut_alpha: float = 1.0
    lidar_ut_beta: float = 2.0
    lidar_ut_kappa: float = 0.0
    # If <=0, use sqrt(D+lambda)
    lidar_ut_delta: float = -1.0
    lidar_ut_in_image_margin_factor: float = 0.1
    lidar_ut_require_all_sigma_points_valid: bool = False
    # Densification / pruning (gsplat strategy)
    densify: DensifyConfig = field(default_factory=DensifyConfig)


@dataclass
class OptimizerConfig:
    lr_mean: float = 1.6e-4
    lr_color: float = 2.5e-3
    lr_opacity: float = 5.0e-2
    lr_scale: float = 5.0e-3
    lr_rotation: float = 1.0e-3
    weight_decay: float = 0.0


@dataclass
class LossConfig:
    rgb_l1_weight: float = 1.0
    # RGB loss mixing: L = (1-λ)*L1 + λ*(1-SSIM)
    rgb_ssim_lambda: float = 0.2
    lidar_depth_weight: float = 1.0
    lidar_loss_type: str = "smooth_l1"
    # LiDAR supervision sampling:
    # - "uniform": treat all valid pixels equally
    # - "angle_table": weight pixels by the LiDAR ring angle table (solid-angle proxy)
    lidar_loss_sampling: str = "uniform"
    lidar_opacity_binarize_weight: float = 0.0
    opacity_reg_weight: float = 1.0e-4
    scale_reg_weight: float = 1.0e-4


@dataclass
class TrainingConfig:
    batch_size: int = 1
    num_workers: int = 0
    max_steps: int = 10000
    log_every: int = 20
    save_every: int = 1000
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"
    train_split_ratio: float = 0.8
    split_seed: int = 0
    # Eval knobs (speed)
    eval_compute_rgb_psnr: bool = False
    eval_compute_lidar_rmse: bool = False
    eval_compute_cov: bool = False
    eval_save_vis: bool = False
    eval_compare_fixed_fitted: bool = False
    eval_compute_lidar_pearson: bool = True

    # Visualization
    vis_every: int = 200
    vis_dir: str = "checkpoints/vis"


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def _construct_dataclass(cls: type[Any], values: dict[str, Any]) -> Any:
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    dataset_raw = dict(raw["dataset"])
    if dataset_raw.get("mode", "manifest") == "manifest" and not dataset_raw.get("manifest_path"):
        raise ValueError("dataset.manifest_path is required when dataset.mode='manifest'.")
    if dataset_raw.get("mode") == "kitti_r":
        for key in ("source_path", "data_seq"):
            if not dataset_raw.get(key):
                raise ValueError(f"dataset.{key} is required when dataset.mode='kitti_r'.")
    if dataset_raw.get("mode") == "waymo":
        for key in ("source_path", "waymo_scene_id"):
            if not dataset_raw.get(key):
                raise ValueError(f"dataset.{key} is required when dataset.mode='waymo'.")
    if dataset_raw.get("mode") == "pandaset":
        for key in ("source_path", "pandaset_sequence_id"):
            if not dataset_raw.get(key):
                raise ValueError(f"dataset.{key} is required when dataset.mode='pandaset'.")

    model_raw = dict(raw.get("model", {}) or {})
    if isinstance(model_raw.get("densify"), dict):
        model_raw["densify"] = _construct_dataclass(DensifyConfig, model_raw["densify"])

    return ExperimentConfig(
        dataset=_construct_dataclass(DatasetConfig, dataset_raw),
        model=_construct_dataclass(ModelConfig, model_raw),
        optimizer=_construct_dataclass(OptimizerConfig, raw.get("optimizer", {})),
        loss=_construct_dataclass(LossConfig, raw.get("loss", {})),
        training=_construct_dataclass(TrainingConfig, raw.get("training", {})),
    )
