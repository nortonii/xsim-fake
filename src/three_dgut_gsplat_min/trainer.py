from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .config import ExperimentConfig
from .data import KittiRDataset, MultiSensorDataset, multi_sensor_collate_fn
from .data_pandaset import PandaSetDataset
from .data_waymo import WaymoDataset
from .lidar_models import HDL64E_VERT_DEG, PANDAR64_ROT_DEG, PANDAR64_VERT_DEG, WAYMO_TOP_ROWS, WAYMO_TOP_VERT_DEG
from .losses import JointLoss
from .model import GaussianSceneModel

try:
    import gsplat
    from gsplat.strategy.ops import remove as gsplat_remove
except Exception:  # pragma: no cover
    gsplat = None
    gsplat_remove = None


class JointTrainer:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")

        if config.dataset.mode in ("kitti_r", "waymo", "pandaset"):
            self._ensure_fixed_vertical_angles(config)

        if config.dataset.mode == "manifest":
            if not config.dataset.manifest_path:
                raise ValueError("dataset.manifest_path must be set when dataset.mode='manifest'.")
            self.dataset = MultiSensorDataset(
                manifest_path=config.dataset.manifest_path,
                rgb_tensor_key=config.dataset.rgb_tensor_key,
                lidar_depth_key=config.dataset.lidar_depth_key,
            )
        elif config.dataset.mode == "kitti_r":
            if not config.dataset.source_path or not config.dataset.data_seq:
                raise ValueError("dataset.source_path and dataset.data_seq must be set when dataset.mode='kitti_r'.")
            self.dataset = KittiRDataset(
                source_path=config.dataset.source_path,
                data_seq=config.dataset.data_seq,
                cam_id=config.dataset.cam_id,
                start_index=config.dataset.start_index,
                data_type=config.dataset.data_type,
                segment_length=config.dataset.segment_length,
                lidar_width=config.dataset.lidar_width,
                lidar_height=config.dataset.lidar_height,
                max_range=config.dataset.max_range,
                near_plane=config.dataset.near_plane,
                far_plane=config.dataset.far_plane,
                lidar_vertical_fov_min_deg=config.dataset.lidar_vertical_fov_min_deg,
                lidar_vertical_fov_max_deg=config.dataset.lidar_vertical_fov_max_deg,
                lidar_vertical_angles_deg=config.dataset.lidar_vertical_angles_deg,
                lidar_vertical_angle_offset_deg=config.dataset.lidar_vertical_angle_offset_deg,
                lidar_angle_mode=config.dataset.lidar_angle_mode,
                lidar_dbxml_path=config.dataset.lidar_dbxml_path,
            )
        elif config.dataset.mode == "waymo":
            if not config.dataset.source_path or not config.dataset.waymo_scene_id:
                raise ValueError("dataset.source_path and dataset.waymo_scene_id must be set when dataset.mode='waymo'.")
            self.dataset = WaymoDataset(
                source_path=config.dataset.source_path,
                scene_id=config.dataset.waymo_scene_id,
                cam_id=config.dataset.cam_id,
                start_index=config.dataset.start_index,
                segment_length=config.dataset.segment_length,
                lidar_width=config.dataset.lidar_width,
                lidar_height=config.dataset.lidar_height,
                max_range=config.dataset.max_range,
                near_plane=config.dataset.near_plane,
                far_plane=config.dataset.far_plane,
                lidar_vertical_fov_min_deg=config.dataset.lidar_vertical_fov_min_deg,
                lidar_vertical_fov_max_deg=config.dataset.lidar_vertical_fov_max_deg,
                lidar_vertical_angles_deg=config.dataset.lidar_vertical_angles_deg,
            )
        elif config.dataset.mode == "pandaset":
            if not config.dataset.source_path or not config.dataset.pandaset_sequence_id:
                raise ValueError(
                    "dataset.source_path and dataset.pandaset_sequence_id must be set when dataset.mode='pandaset'."
                )
            self.dataset = PandaSetDataset(
                source_path=config.dataset.source_path,
                sequence_id=config.dataset.pandaset_sequence_id,
                camera_name=config.dataset.pandaset_camera_name,
                lidar_sensor_id=config.dataset.pandaset_lidar_sensor_id,
                start_index=config.dataset.start_index,
                segment_length=config.dataset.segment_length,
                lidar_width=config.dataset.lidar_width,
                lidar_height=config.dataset.lidar_height,
                max_range=config.dataset.max_range,
                near_plane=config.dataset.near_plane,
                far_plane=config.dataset.far_plane,
                lidar_vertical_fov_min_deg=config.dataset.lidar_vertical_fov_min_deg,
                lidar_vertical_fov_max_deg=config.dataset.lidar_vertical_fov_max_deg,
                lidar_vertical_angles_deg=config.dataset.lidar_vertical_angles_deg,
                lidar_row_azimuth_offsets_deg=config.dataset.lidar_row_azimuth_offsets_deg,
                lidar_vertical_angle_offset_deg=config.dataset.lidar_vertical_angle_offset_deg,
                lidar_angle_mode=config.dataset.lidar_angle_mode,
            )
        else:
            raise ValueError(f"Unsupported dataset.mode: {config.dataset.mode!r}")

        # Mirror dataset-fitted vertical angles back into the config so that
        # rendering, loss weighting, and eval all use the same sampling map.
        fitted_angles = getattr(self.dataset, "lidar_vertical_angles_deg", None)
        if fitted_angles is not None and len(fitted_angles) > 0:
            self.config.dataset.lidar_vertical_angles_deg = list(fitted_angles)
        fitted_azimuth_offsets = getattr(self.dataset, "lidar_row_azimuth_offsets_deg", None)
        if fitted_azimuth_offsets is not None and len(fitted_azimuth_offsets) > 0:
            self.config.dataset.lidar_row_azimuth_offsets_deg = list(fitted_azimuth_offsets)

        # Train/test split (deterministic)
        n_total = len(self.dataset)
        n_train = int(n_total * float(config.training.train_split_ratio))
        n_train = max(1, min(n_total - 1, n_train))
        rng = np.random.default_rng(int(config.training.split_seed))
        indices = np.arange(n_total)
        rng.shuffle(indices)
        train_idx = indices[:n_train].tolist()
        test_idx = indices[n_train:].tolist()

        self.train_set = Subset(self.dataset, train_idx)
        self.test_set = Subset(self.dataset, test_idx)

        self.loader = DataLoader(
            self.train_set,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            collate_fn=multi_sensor_collate_fn,
        )
        # Test evaluation loader (deterministic order).
        self.eval_loader = DataLoader(
            self.test_set,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=multi_sensor_collate_fn,
        )

        self.model = GaussianSceneModel(
            num_gaussians=config.model.num_gaussians,
            sh_degree=config.model.sh_degree,
            init_extent=config.model.init_extent,
            init_opacity=config.model.init_opacity,
            init_scale=config.model.init_scale,
            background_color=config.model.background_color,
            use_separate_opacity=config.model.use_separate_opacity,
            lidar_ut_enable=config.model.lidar_ut_enable,
            lidar_ut_alpha=config.model.lidar_ut_alpha,
            lidar_ut_beta=config.model.lidar_ut_beta,
            lidar_ut_kappa=config.model.lidar_ut_kappa,
            lidar_ut_delta=config.model.lidar_ut_delta,
            lidar_ut_in_image_margin_factor=config.model.lidar_ut_in_image_margin_factor,
            lidar_ut_require_all_sigma_points_valid=config.model.lidar_ut_require_all_sigma_points_valid,
        ).to(self.device)
        self.model.configure_lidar_model(config.dataset)

        # gsplat densification strategies expect a single opacity tensor.
        # If dual-opacity is enabled, we will feed camera opacity to the strategy.
        # Recommended: initialize gaussians from aggregated LiDAR point clouds.
        # This matches the common practice in HiGS-Calib and avoids starting from an
        # empty/irrelevant region (which makes pred_rgb look black).
        if config.dataset.mode in ("kitti_r", "waymo", "pandaset"):
            with torch.no_grad():
                n = int(self.model.means.shape[0])

                means_w: torch.Tensor | None = None
                if getattr(config.model, "init_from_lidar", True):
                    frames = int(getattr(config.model, "init_lidar_frames", 20))
                    frames = max(1, min(frames, len(self.dataset)))

                    pts_world: list[np.ndarray] = []
                    for i in range(frames):
                        sample_i = self.dataset[i]
                        if sample_i.lidar_points is None or sample_i.lidar_points.numel() == 0:
                            continue
                        xyz_l = sample_i.lidar_points.detach().float().cpu().numpy()
                        T_w_l = sample_i.lidar_to_world.detach().float().cpu().numpy()
                        ones = np.ones((xyz_l.shape[0], 1), dtype=np.float32)
                        xyz_h = np.concatenate([xyz_l.astype(np.float32), ones], axis=1)
                        xyz_w = (T_w_l @ xyz_h.T).T[:, :3]
                        pts_world.append(xyz_w)

                    if pts_world:
                        all_pts = np.concatenate(pts_world, axis=0)

                        # Export the aggregated initialization point cloud for visualization (optional).
                        if getattr(config.model, "save_init_ply", False):
                            try:
                                init_dir = Path(config.training.checkpoint_dir) / "geometry"
                                init_dir.mkdir(parents=True, exist_ok=True)
                                scene_tag = str(
                                    getattr(self.dataset, "scene_name", getattr(self.dataset, "scene_id", config.dataset.mode))
                                )
                                ply_path = init_dir / f"init_points_world_{scene_tag}_frames{frames}.ply"

                                # Random downsample for file size.
                                max_vis = 2_000_000
                                vis_pts = all_pts
                                if vis_pts.shape[0] > max_vis:
                                    vidx = np.random.choice(vis_pts.shape[0], size=max_vis, replace=False)
                                    vis_pts = vis_pts[vidx]

                                with ply_path.open("w", encoding="utf-8") as f:
                                    f.write("ply\nformat ascii 1.0\n")
                                    f.write(f"element vertex {vis_pts.shape[0]}\n")
                                    f.write("property float x\nproperty float y\nproperty float z\n")
                                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                                    f.write("end_header\n")
                                    for x, y, z in vis_pts.astype(np.float32):
                                        f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 255 255\n")
                            except Exception:
                                pass

                        if all_pts.shape[0] > 0:
                            # Sample exactly n points (with replacement if needed)
                            if all_pts.shape[0] >= n:
                                idx = np.random.choice(all_pts.shape[0], size=n, replace=False)
                            else:
                                idx = np.random.choice(all_pts.shape[0], size=n, replace=True)
                            means_w = torch.from_numpy(all_pts[idx]).to(self.device, dtype=self.model.means.dtype)

                if means_w is None:
                    # Fallback: initialize in front of the first camera so RGB rendering is non-degenerate.
                    sample0 = self.dataset[0]
                    cam_to_world = sample0.camera_to_world.to(self.device)

                    extent = float(config.model.init_extent)
                    means_c = (torch.rand((n, 3), device=self.device, dtype=self.model.means.dtype) - 0.5) * (2.0 * extent)
                    means_c[:, 2] = torch.rand((n,), device=self.device, dtype=self.model.means.dtype) * (2.0 * extent - 1.0) + 1.0

                    ones_t = torch.ones((n, 1), device=self.device, dtype=self.model.means.dtype)
                    means_c_h = torch.cat([means_c, ones_t], dim=-1)
                    means_w = (cam_to_world @ means_c_h.T).T[:, :3]

                self.model.means.copy_(means_w)

                # Export initialized Gaussian means for visualization (optional).
                if getattr(config.model, "save_init_ply", False):
                    try:
                        init_dir = Path(config.training.checkpoint_dir) / "geometry"
                        init_dir.mkdir(parents=True, exist_ok=True)
                        means_np = self.model.means.detach().float().cpu().numpy()
                        scene_tag = str(
                            getattr(self.dataset, "scene_name", getattr(self.dataset, "scene_id", config.dataset.mode))
                        )
                        ply_path = init_dir / f"gaussians_init_means_{scene_tag}.ply"
                        with ply_path.open("w", encoding="utf-8") as f:
                            f.write("ply\nformat ascii 1.0\n")
                            f.write(f"element vertex {means_np.shape[0]}\n")
                            f.write("property float x\nproperty float y\nproperty float z\n")
                            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                            f.write("end_header\n")
                            for x, y, z in means_np.astype(np.float32):
                                f.write(f"{x:.6f} {y:.6f} {z:.6f} 255 0 255\n")
                    except Exception:
                        pass

        self.criterion = JointLoss(config.loss, config.dataset)

        # --- Optimizers: one-parameter-per-optimizer (required by gsplat strategies) ---
        # 3DGUT / XSIM representation: if densify is enabled, use camera opacity as strategy opacity.

        self.params: dict[str, torch.nn.Parameter] = {
            "means": self.model.means,
            "sh_coeffs": self.model.sh_coeffs,
            "scales": self.model.scales,
            "quats": self.model.quats,
        }
        if self.model.use_separate_opacity:
            self.params["opacity_camera"] = self.model.opacity_camera
            self.params["opacity_lidar"] = self.model.opacity_lidar
            if config.model.densify.enable:
                # Alias for gsplat strategy
                self.params["opacities"] = self.params["opacity_camera"]
        else:
            self.params["opacities"] = self.model.opacities

        self.optimizers: dict[str, torch.optim.Optimizer] = {
            "means": torch.optim.Adam([self.params["means"]], lr=config.optimizer.lr_mean, weight_decay=config.optimizer.weight_decay),
            "sh_coeffs": torch.optim.Adam([self.params["sh_coeffs"]], lr=config.optimizer.lr_color, weight_decay=config.optimizer.weight_decay),
            "scales": torch.optim.Adam([self.params["scales"]], lr=config.optimizer.lr_scale, weight_decay=config.optimizer.weight_decay),
            "quats": torch.optim.Adam([self.params["quats"]], lr=config.optimizer.lr_rotation, weight_decay=config.optimizer.weight_decay),
        }
        if self.model.use_separate_opacity:
            self.optimizers["opacity_camera"] = torch.optim.Adam([self.params["opacity_camera"]], lr=config.optimizer.lr_opacity, weight_decay=config.optimizer.weight_decay)
            self.optimizers["opacity_lidar"] = torch.optim.Adam([self.params["opacity_lidar"]], lr=config.optimizer.lr_opacity, weight_decay=config.optimizer.weight_decay)
            if config.model.densify.enable:
                # Alias for gsplat strategy to step camera opacity
                self.optimizers["opacities"] = self.optimizers["opacity_camera"]
        else:
            self.optimizers["opacities"] = torch.optim.Adam([self.params["opacities"]], lr=config.optimizer.lr_opacity, weight_decay=config.optimizer.weight_decay)

        # --- gsplat strategy for densification/pruning ---
        self.strategy = None
        self.strategy_state = None
        self.densify_cfg = None
        if getattr(config.model, "densify", None) is not None and config.model.densify.enable:
            self.densify_cfg = config.model.densify
            if gsplat is None:
                raise ImportError("gsplat is required when model.densify.enable=True")

            if config.model.densify.strategy == "default":
                self.strategy = gsplat.DefaultStrategy(
                    prune_opa=config.model.densify.prune_opa,
                    grow_grad2d=config.model.densify.grow_grad2d,
                    grow_scale3d=config.model.densify.grow_scale3d,
                    grow_scale2d=config.model.densify.grow_scale2d,
                    prune_scale3d=config.model.densify.prune_scale3d,
                    prune_scale2d=config.model.densify.prune_scale2d,
                    refine_scale2d_stop_iter=config.model.densify.refine_scale2d_stop_iter,
                    refine_start_iter=config.model.densify.refine_start_iter,
                    refine_stop_iter=config.model.densify.refine_stop_iter,
                    reset_every=config.model.densify.reset_every,
                    refine_every=config.model.densify.refine_every,
                    pause_refine_after_reset=config.model.densify.pause_refine_after_reset,
                    absgrad=config.model.densify.absgrad,
                    revised_opacity=config.model.densify.revised_opacity,
                    verbose=config.model.densify.verbose,
                    key_for_gradient=config.model.densify.key_for_gradient,
                )
            elif config.model.densify.strategy == "mcmc":
                # Note: gsplat 1.5.3 MCMCStrategy has a different set of hyperparams.
                self.strategy = gsplat.MCMCStrategy(
                    refine_start_iter=config.model.densify.refine_start_iter,
                    refine_stop_iter=config.model.densify.refine_stop_iter,
                    refine_every=config.model.densify.refine_every,
                    min_opacity=config.model.densify.prune_opa,
                    verbose=config.model.densify.verbose,
                )
            else:
                raise ValueError(f"Unsupported densify strategy: {config.model.densify.strategy!r}")

            if isinstance(self.strategy, gsplat.DefaultStrategy):
                if config.model.densify.scene_scale is not None:
                    scene_scale = float(config.model.densify.scene_scale)
                else:
                    scene_scale = float(getattr(config.dataset, "max_range", 1.0) or 1.0)
                self.strategy_state = self.strategy.initialize_state(scene_scale=scene_scale)
            else:
                # MCMCStrategy.initialize_state() takes no args in gsplat 1.5.3
                self.strategy_state = self.strategy.initialize_state()

            # Make model render return abs-grad info when requested by strategy
            setattr(self.model, "_strategy_absgrad", bool(config.model.densify.absgrad))

        # Defer bad-Gaussian pruning to a safe step boundary.
        self._pending_bad_gaussian_ids: set[int] = set()

        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.vis_dir = Path(config.training.vis_dir)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self._log_projection_sanity()

    @staticmethod
    def _ensure_fixed_vertical_angles(config: ExperimentConfig) -> None:
        """Populate a deterministic fixed angle table for lidar rendering/eval.

        The native gsplat lidar backend expects an explicit per-row angle table.
        Prefer built-in sensor tables when they are known; otherwise synthesize a
        linear table once so the dataset and renderer share the same row semantics.
        """
        if config.dataset.lidar_angle_mode.strip().lower() != "fixed":
            return
        sensor = str(getattr(config.dataset, "lidar_sensor", "") or "").strip().lower()
        dataset_mode = str(getattr(config.dataset, "mode", "") or "").strip().lower()
        if config.dataset.lidar_vertical_angles_deg is not None and len(config.dataset.lidar_vertical_angles_deg) > 0:
            base_angles = list(config.dataset.lidar_vertical_angles_deg)
        else:
            h = int(config.dataset.lidar_height)
            if h <= 0:
                raise ValueError(f"lidar_height must be positive, got {h}")
            if sensor in {"waymo", "waymo_top"} or dataset_mode == "waymo":
                if h != WAYMO_TOP_ROWS:
                    raise ValueError(
                        f"Waymo fixed angle table expects lidar_height={WAYMO_TOP_ROWS}, got {h}"
                    )
                base_angles = list(WAYMO_TOP_VERT_DEG)
            elif sensor == "hdl64e" and dataset_mode == "kitti_r":
                base_angles = list(HDL64E_VERT_DEG)
            elif sensor == "pandar64" and dataset_mode == "pandaset":
                base_angles = list(PANDAR64_VERT_DEG)
                config.dataset.lidar_row_azimuth_offsets_deg = list(PANDAR64_ROT_DEG)
            else:
                lo = float(config.dataset.lidar_vertical_fov_max_deg)
                hi = float(config.dataset.lidar_vertical_fov_min_deg)
                base_angles = np.linspace(lo, hi, h, dtype=np.float32).tolist()

        interp_factor = int(getattr(config.dataset, "lidar_vertical_angles_interp_factor", 1))
        if interp_factor > 1 and len(base_angles) > 1:
            angles = JointTrainer._upsample_vertical_angles(base_angles, interp_factor)
            config.dataset.lidar_vertical_angles_deg = angles
            config.dataset.lidar_height = len(angles)
            return

        config.dataset.lidar_vertical_angles_deg = base_angles

    @staticmethod
    def _upsample_vertical_angles(angles_deg: list[float], factor: int) -> list[float]:
        """Linearly densify a monotonic angle table while preserving original samples."""
        if factor <= 1 or len(angles_deg) <= 1:
            return [float(v) for v in angles_deg]
        sorted_angles = np.asarray(sorted((float(v) for v in angles_deg), reverse=True), dtype=np.float32)
        upsampled: list[float] = [float(sorted_angles[0])]
        for idx in range(len(sorted_angles) - 1):
            a0 = float(sorted_angles[idx])
            a1 = float(sorted_angles[idx + 1])
            for step in range(1, factor):
                t = step / float(factor)
                upsampled.append((1.0 - t) * a0 + t * a1)
            upsampled.append(a1)
        return upsampled

    def _sync_opacity_params(self) -> None:
        """Ensure opacity tensors match current gaussian count after densify."""
        if not self.model.use_separate_opacity:
            return
        n_means = int(self.model.means.shape[0])
        n_opa = int(self.model.opacity_camera.shape[0])
        if n_means == n_opa:
            return
        with torch.no_grad():
            init_opa = float(self.config.model.init_opacity)
            init_logit = float(self.model._inverse_sigmoid(init_opa))
            device = self.model.opacity_camera.device
            dtype = self.model.opacity_camera.dtype
            if n_means > n_opa:
                pad = n_means - n_opa
                pad_vals = torch.full((pad, 1), init_logit, device=device, dtype=dtype)
                new_cam = torch.cat([self.model.opacity_camera, pad_vals], dim=0)
                new_lidar = torch.cat([self.model.opacity_lidar, pad_vals], dim=0)
            else:
                new_cam = self.model.opacity_camera[:n_means].clone()
                new_lidar = self.model.opacity_lidar[:n_means].clone()
        self.model.opacity_camera = torch.nn.Parameter(new_cam)
        self.model.opacity_lidar = torch.nn.Parameter(new_lidar)
        self.params["opacity_camera"] = self.model.opacity_camera
        self.params["opacity_lidar"] = self.model.opacity_lidar
        if self.config.model.densify.enable:
            self.params["opacities"] = self.params["opacity_camera"]
        self.optimizers["opacity_camera"] = torch.optim.Adam(
            [self.params["opacity_camera"]],
            lr=self.config.optimizer.lr_opacity,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.optimizers["opacity_lidar"] = torch.optim.Adam(
            [self.params["opacity_lidar"]],
            lr=self.config.optimizer.lr_opacity,
            weight_decay=self.config.optimizer.weight_decay,
        )
        if self.config.model.densify.enable:
            self.optimizers["opacities"] = self.optimizers["opacity_camera"]

    def _sanitize_opacity_params(self, where: str) -> None:
        """Replace non-finite opacity logits with a finite default so training can continue."""
        init_logit = float(self.model._inverse_sigmoid(float(self.config.model.init_opacity)))
        init_log_scale = float(np.log(max(float(self.config.model.init_scale), 1.0e-6)))
        if self.model.use_separate_opacity:
            targets: list[tuple[str, torch.nn.Parameter]] = [
                ("opacity_camera", self.params["opacity_camera"]),
                ("opacity_lidar", self.params["opacity_lidar"]),
            ]
        else:
            targets = [("opacities", self.params["opacities"])]
        targets.append(("scales", self.params["scales"]))

        for name, param in targets:
            data = param.data
            bad = ~torch.isfinite(data)
            if bool(bad.any()):
                n_bad = int(bad.sum().item())
                if name == "scales":
                    fill_value = init_log_scale
                else:
                    fill_value = init_logit
                print(
                    f"[opacity-fix] {where}: repaired {n_bad} non-finite logits in {name} "
                    f"(fill_value={fill_value:.6f})",
                    flush=True,
                )
                data[bad] = fill_value

    def _sanitize_opacity_grads(self, where: str) -> None:
        """Zero any non-finite opacity gradients before they reach the optimizer."""
        init_log_scale = float(np.log(max(float(self.config.model.init_scale), 1.0e-6)))
        if self.model.use_separate_opacity:
            targets: list[tuple[str, torch.nn.Parameter]] = [
                ("opacity_camera", self.params["opacity_camera"]),
                ("opacity_lidar", self.params["opacity_lidar"]),
            ]
        else:
            targets = [("opacities", self.params["opacities"])]
        targets.append(("scales", self.params["scales"]))

        for name, param in targets:
            grad = param.grad
            if grad is None:
                continue
            bad = ~torch.isfinite(grad)
            if bool(bad.any()):
                n_bad = int(bad.sum().item())
                print(
                    f"[opacity-fix] {where}: zeroed {n_bad} non-finite grads in {name}",
                    flush=True,
                )
                grad.data[bad] = 0.0

    def _sanitize_opacity_optimizer_state(self, where: str) -> None:
        """Zero non-finite Adam state for opacity parameters."""
        if self.model.use_separate_opacity:
            names = ("opacity_camera", "opacity_lidar")
        else:
            names = ("opacities",)
        names = names + ("scales",)

        for name in names:
            opt = self.optimizers.get(name)
            param = self.params.get(name)
            if opt is None or param is None:
                continue
            state = opt.state.get(param, {})
            for key, value in state.items():
                if not isinstance(value, torch.Tensor):
                    continue
                bad = ~torch.isfinite(value)
                if bool(bad.any()):
                    n_bad = int(bad.sum().item())
                    print(
                        f"[opacity-fix] {where}: zeroed {n_bad} non-finite state values in {name}.{key}",
                        flush=True,
                    )
                    value.data[bad] = 0.0

    @staticmethod
    def _tensor_finite_summary(x: torch.Tensor) -> str:
        x = x.detach()
        finite = torch.isfinite(x)
        n = int(x.numel())
        n_finite = int(finite.sum().item())
        if n_finite > 0:
            x_finite = x[finite]
            x_min = float(x_finite.min().item())
            x_max = float(x_finite.max().item())
            x_mean = float(x_finite.mean().item())
        else:
            x_min = x_max = x_mean = float("nan")
        return f"shape={tuple(x.shape)} finite={n_finite}/{n} min={x_min:.6e} max={x_max:.6e} mean={x_mean:.6e}"

    @staticmethod
    def _tensor_values_summary(x: torch.Tensor, max_items: int = 8) -> str:
        x = x.detach().float().cpu().reshape(-1)
        vals = [float(v) for v in x[:max_items].tolist()]
        suffix = "" if x.numel() <= max_items else ", ..."
        return "[" + ", ".join(f"{v:.6e}" for v in vals) + suffix + "]"

    def _check_tensor_finite(self, name: str, x: torch.Tensor, step: int) -> None:
        if torch.isfinite(x).all():
            return
        print(f"[nan-probe] step={step} {name} {self._tensor_finite_summary(x)}", flush=True)
        raise FloatingPointError(f"Non-finite tensor detected in {name} at step={step}")

    def _dump_gaussian_context(
        self,
        step: int,
        term_name: str,
        mon_name: str,
        bad_idx: torch.Tensor,
        grad: torch.Tensor,
        batch: dict[str, object] | None = None,
    ) -> None:
        if bad_idx.ndim == 0:
            row_ids = [int(bad_idx.item())]
        elif bad_idx.ndim == 1:
            row_ids = sorted({int(i) for i in bad_idx.detach().flatten().tolist()})
        else:
            row_ids = sorted({int(i) for i in bad_idx[:, 0].detach().flatten().tolist()})
        if not row_ids:
            return

        # Keep the dump short while still exposing the exact offending row.
        row_ids = row_ids[:4]
        print(
            f"[term-probe] step={step} term={term_name} mon={mon_name} offending_gaussians={row_ids}",
            flush=True,
        )

        def _safe_row(t: torch.Tensor, idx: int) -> torch.Tensor:
            row = t[idx].detach().float()
            return row.reshape(-1)

        lidar_to_world = None
        if batch is not None:
            maybe_l2w = batch.get("lidar_to_world", None)
            if torch.is_tensor(maybe_l2w):
                lidar_to_world = maybe_l2w.detach().float()
                if lidar_to_world.ndim == 3:
                    lidar_to_world = lidar_to_world[0]
            elif isinstance(maybe_l2w, (list, tuple)) and maybe_l2w and torch.is_tensor(maybe_l2w[0]):
                lidar_to_world = maybe_l2w[0].detach().float()

        for gid in row_ids:
            mean = _safe_row(self.params["means"], gid)
            log_scale = _safe_row(self.params["scales"], gid)
            scale = torch.exp(log_scale).clamp_min(1.0e-6)
            quat = _safe_row(self.params["quats"], gid)
            quat_norm = float(torch.linalg.vector_norm(quat).item())
            sh0 = _safe_row(self.params["sh_coeffs"][:, 0, :], gid)
            lidar_xyz_str = "lidar_xyz=n/a"
            lidar_r_str = "lidar_r=n/a"
            if lidar_to_world is not None:
                mean_h = torch.cat([mean, torch.ones(1, device=mean.device, dtype=mean.dtype)], dim=0)
                mean_l = torch.linalg.inv(lidar_to_world.to(device=mean.device, dtype=mean.dtype)) @ mean_h
                mean_l = mean_l[:3]
                lidar_xyz_str = f"lidar_xyz={self._tensor_values_summary(mean_l)}"
                lidar_r_str = f"lidar_r={float(torch.linalg.vector_norm(mean_l).item()):.6e}"
            if self.model.use_separate_opacity:
                opa_cam = float(torch.sigmoid(torch.nan_to_num(self.params["opacity_camera"][gid].detach().float(), nan=0.0, posinf=20.0, neginf=-20.0)).item())
                opa_lidar = float(torch.sigmoid(torch.nan_to_num(self.params["opacity_lidar"][gid].detach().float(), nan=0.0, posinf=20.0, neginf=-20.0)).item())
                opa_str = f"opacity_camera={opa_cam:.6e} opacity_lidar={opa_lidar:.6e}"
            else:
                opa = float(torch.sigmoid(torch.nan_to_num(self.params["opacities"][gid].detach().float(), nan=0.0, posinf=20.0, neginf=-20.0)).item())
                opa_str = f"opacities={opa:.6e}"
            grad_row = _safe_row(grad, gid)
            mean_norm = float(torch.linalg.vector_norm(mean).item())
            scale_norm = float(torch.linalg.vector_norm(scale).item())
            print(
                f"[term-probe] step={step} term={term_name} mon={mon_name} gaussian={gid} "
                f"mean={self._tensor_values_summary(mean)} mean_norm={mean_norm:.6e} "
                f"log_scale={self._tensor_values_summary(log_scale)} scale={self._tensor_values_summary(scale)} "
                f"scale_norm={scale_norm:.6e} quat={self._tensor_values_summary(quat)} quat_norm={quat_norm:.6e} "
                f"sh0={self._tensor_values_summary(sh0)} {opa_str} {lidar_xyz_str} {lidar_r_str} "
                f"grad_row={self._tensor_values_summary(grad_row)}",
                flush=True,
            )

    def _prune_bad_gaussians(
        self,
        step: int,
        term_name: str,
        mon_name: str,
        bad_idx: torch.Tensor,
    ) -> int:
        if gsplat_remove is None:
            raise ImportError("gsplat.strategy.ops.remove is required for bad-gaussian pruning.")

        if bad_idx.ndim == 0:
            row_ids = torch.unique(bad_idx.reshape(1).long())
        elif bad_idx.ndim == 1:
            row_ids = torch.unique(bad_idx.long())
        else:
            row_ids = torch.unique(bad_idx[:, 0].long())
        if row_ids.numel() == 0:
            return 0
        row_ids = row_ids[(row_ids >= 0) & (row_ids < self.params["means"].shape[0])]
        if row_ids.numel() == 0:
            return 0

        mask = torch.zeros((int(self.params["means"].shape[0]),), device=self.params["means"].device, dtype=torch.bool)
        mask[row_ids] = True

        # gsplat remove() cannot safely handle alias keys that point to the same underlying
        # Parameter object. Temporarily drop the alias and restore it after pruning.
        alias_removed = False
        saved_opacity_opt = None
        if self.model.use_separate_opacity and "opacities" in self.params and self.params.get("opacities") is self.params.get("opacity_camera"):
            self.params.pop("opacities", None)
            saved_opacity_opt = self.optimizers.pop("opacities", None)
            alias_removed = True

        state = self.strategy_state if self.strategy_state is not None else {}
        gsplat_remove(self.params, self.optimizers, state, mask)

        if alias_removed:
            self.params["opacities"] = self.params["opacity_camera"]
            if saved_opacity_opt is not None:
                self.optimizers["opacities"] = self.optimizers["opacity_camera"]

        # Sync the nn.Module fields to the updated parameter objects.
        for k in ("means", "sh_coeffs", "scales", "quats", "opacities"):
            if k in self.params and hasattr(self.model, k):
                if getattr(self.model, k) is not self.params[k]:
                    setattr(self.model, k, self.params[k])
        if self.model.use_separate_opacity:
            if self.model.opacity_camera is not self.params["opacity_camera"]:
                self.model.opacity_camera = self.params["opacity_camera"]
            if self.model.opacity_lidar is not self.params["opacity_lidar"]:
                self.model.opacity_lidar = self.params["opacity_lidar"]

        print(
            f"[term-probe] step={step} term={term_name} mon={mon_name} pruned_bad_gaussians="
            f"{row_ids[:16].tolist()} n_pruned={int(row_ids.numel())}",
            flush=True,
        )
        self._sanitize_opacity_params("post_prune")
        self._sanitize_opacity_optimizer_state("post_prune")
        return int(row_ids.numel())

    def _queue_bad_gaussians(self, bad_idx: torch.Tensor) -> list[int]:
        if bad_idx.ndim == 0:
            row_ids = [int(bad_idx.item())]
        elif bad_idx.ndim == 1:
            row_ids = sorted({int(i) for i in bad_idx.detach().flatten().tolist()})
        else:
            row_ids = sorted({int(i) for i in bad_idx[:, 0].detach().flatten().tolist()})
        row_ids = [rid for rid in row_ids if 0 <= rid < int(self.params["means"].shape[0])]
        if row_ids:
            self._pending_bad_gaussian_ids.update(row_ids)
        return row_ids

    def _apply_pending_bad_gaussian_prune(self, step: int, where: str) -> int:
        if not self._pending_bad_gaussian_ids:
            return 0
        if gsplat_remove is None:
            raise ImportError("gsplat.strategy.ops.remove is required for bad-gaussian pruning.")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        row_ids = sorted(
            rid for rid in self._pending_bad_gaussian_ids if 0 <= rid < int(self.params["means"].shape[0])
        )
        self._pending_bad_gaussian_ids.clear()
        if not row_ids:
            return 0

        mask = torch.zeros((int(self.params["means"].shape[0]),), device=self.params["means"].device, dtype=torch.bool)
        mask[row_ids] = True

        alias_removed = False
        saved_opacity_opt = None
        if self.model.use_separate_opacity and "opacities" in self.params and self.params.get("opacities") is self.params.get("opacity_camera"):
            self.params.pop("opacities", None)
            saved_opacity_opt = self.optimizers.pop("opacities", None)
            alias_removed = True

        state = self.strategy_state if self.strategy_state is not None else {}
        gsplat_remove(self.params, self.optimizers, state, mask)

        if alias_removed:
            self.params["opacities"] = self.params["opacity_camera"]
            if saved_opacity_opt is not None:
                self.optimizers["opacities"] = self.optimizers["opacity_camera"]

        for k in ("means", "sh_coeffs", "scales", "quats", "opacities"):
            if k in self.params and hasattr(self.model, k):
                if getattr(self.model, k) is not self.params[k]:
                    setattr(self.model, k, self.params[k])
        if self.model.use_separate_opacity:
            if self.model.opacity_camera is not self.params["opacity_camera"]:
                self.model.opacity_camera = self.params["opacity_camera"]
            if self.model.opacity_lidar is not self.params["opacity_lidar"]:
                self.model.opacity_lidar = self.params["opacity_lidar"]

        self._sanitize_opacity_params("post_pending_prune")
        self._sanitize_opacity_optimizer_state("post_pending_prune")
        print(
            f"[term-probe] step={step} where={where} pruned_pending_bad_gaussians={row_ids[:16]} "
            f"n_pruned={len(row_ids)}",
            flush=True,
        )
        return len(row_ids)

    def _probe_loss_term_grads(
        self,
        step: int,
        loss_terms: dict[str, torch.Tensor],
        batch: dict[str, object],
        render_output: object,
    ) -> bool:
        """Backprop each loss term separately to identify which one first produces bad grads."""
        probe_terms = [
            "rgb_l1",
            "rgb_ssim",
            "lidar_depth",
            "opacity_reg",
            "lidar_opacity_binarize",
            "scale_reg",
        ]
        monitored = []
        if "scales" in self.params:
            monitored.append("scales")
        if "opacities" in self.params:
            monitored.append("opacities")

        def _bad_count(t: torch.Tensor | None) -> int:
            if t is None:
                return 0
            return int((~torch.isfinite(t)).sum().item())

        frame_id = batch.get("frame_id", "unknown")
        if isinstance(frame_id, (list, tuple)) and len(frame_id) == 1:
            frame_id = frame_id[0]
        if not isinstance(frame_id, str):
            frame_id = str(frame_id)

        lidar_pred = getattr(render_output, "lidar").depth
        lidar_tgt = batch.get("lidar_depth", None)
        if torch.is_tensor(lidar_pred) and torch.is_tensor(lidar_tgt):
            valid = lidar_tgt > 0
            finite_pred = torch.isfinite(lidar_pred)
            finite_tgt = torch.isfinite(lidar_tgt)
            print(
                f"[term-probe] step={step} frame={frame_id} "
                f"lidar_pred={self._tensor_finite_summary(lidar_pred)} "
                f"lidar_tgt={self._tensor_finite_summary(lidar_tgt)} "
                f"valid={int(valid.sum().item())} finite_pred={int(finite_pred.sum().item())} "
                f"finite_tgt={int(finite_tgt.sum().item())}",
                flush=True,
            )

        for idx, name in enumerate(probe_terms):
            term = loss_terms.get(name)
            if term is None:
                continue
            if not torch.is_tensor(term):
                continue
            if not term.requires_grad:
                continue
            if not torch.isfinite(term).all():
                print(f"[term-probe] step={step} term={name} value={float(term.detach().item())}", flush=True)
                continue

            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            term.backward(retain_graph=True)

            bad_parts: list[str] = []
            bad_entries: list[tuple[str, torch.Tensor]] = []
            for mon_name in monitored:
                grad = self.params[mon_name].grad
                if grad is None:
                    continue
                n_bad = _bad_count(grad)
                if n_bad > 0:
                    bad_idx = (~torch.isfinite(grad)).nonzero(as_tuple=False)
                    bad_idx_str = ",".join(
                        f"{int(r)},{int(c)}" for r, c in bad_idx[:16].tolist()
                    )
                    bad_parts.append(f"{mon_name}:{n_bad}")
                    print(
                        f"[term-probe] step={step} term={name} mon={mon_name} bad_idx={bad_idx_str} "
                        f"grad={self._tensor_finite_summary(grad)}",
                        flush=True,
                    )
                    bad_entries.append((mon_name, bad_idx))
            if bad_parts:
                msg = ", ".join(bad_parts)
                for mon_name, bad_idx in bad_entries:
                    grad = self.params[mon_name].grad
                    if grad is not None:
                        self._dump_gaussian_context(step, name, mon_name, bad_idx, grad, batch=batch)
                prune_bad = os.getenv("XSIM_PRUNE_BAD_GAUSSIANS", "").strip().lower() in {"1", "true", "yes", "on"}
                if prune_bad:
                    total_queued = 0
                    for mon_name, bad_idx in bad_entries:
                        row_ids = self._queue_bad_gaussians(bad_idx)
                        total_queued += len(row_ids)
                        if row_ids:
                            grad = self.params[mon_name].grad
                            if grad is not None:
                                if grad.ndim == 1:
                                    grad[row_ids] = 0.0
                                else:
                                    grad[row_ids, ...] = 0.0
                    print(
                        f"[term-probe] step={step} term={name} queued_prune_total={total_queued} "
                        f"after_bad_grads={msg}",
                        flush=True,
                    )
                    continue
                print(
                    f"[term-probe] step={step} term={name} bad_grads={msg} "
                    f"scales={self._tensor_finite_summary(self.params['scales'].data)} "
                    f"opacities={self._tensor_finite_summary(self.params['opacities'].data)}",
                    flush=True,
                )
                raise FloatingPointError(f"Bad grads produced by term={name} at step={step}: {msg}")

        for opt in self.optimizers.values():
            opt.zero_grad(set_to_none=True)
        return False

    def _log_projection_sanity(self) -> None:
        """Print a quick geometric sanity metric for lidar->camera projection."""
        try:
            sample = self.dataset[0]
            if sample.lidar_points is None or sample.lidar_points.numel() == 0:
                print("[proj_sanity] skipped (no lidar_points)", flush=True)
                return

            pts_l = sample.lidar_points.detach().float()
            T_wc = sample.camera_to_world.detach().float()
            T_wl = sample.lidar_to_world.detach().float()
            K = sample.intrinsics.detach().float()
            H = int(sample.rgb_height)
            W = int(sample.rgb_width)

            T_cw = torch.linalg.inv(T_wc)
            T_cl = T_cw @ T_wl
            ones = torch.ones((pts_l.shape[0], 1), dtype=torch.float32)
            pts_l_h = torch.cat([pts_l, ones], dim=-1)
            pts_c = (T_cl @ pts_l_h.T).T[:, :3]

            z = pts_c[:, 2]
            front = z > 1.0e-6
            if not bool(front.any()):
                print("[proj_sanity] WARNING: no lidar points in front of camera (z>0)", flush=True)
                return

            x = pts_c[front, 0]
            y = pts_c[front, 1]
            zf = pts_c[front, 2]
            u = K[0, 0] * (x / zf) + K[0, 2]
            v = K[1, 1] * (y / zf) + K[1, 2]
            inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)

            front_ratio = float(front.float().mean().item())
            inb_ratio_front = float(inb.float().mean().item())
            inb_ratio_all = float((front.float().mean() * inb.float().mean()).item())
            print(
                f"[proj_sanity] frame={sample.frame_id} front_ratio={front_ratio:.3f} "
                f"inb_ratio_front={inb_ratio_front:.3f} inb_ratio_all={inb_ratio_all:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[proj_sanity] skipped due to error: {exc}", flush=True)

    def train(self) -> None:
        step = 0
        self.model.train()
        while step < self.config.training.max_steps:
            for batch in self.loader:
                step += 1
                if step > self.config.training.max_steps:
                    break

                self._apply_pending_bad_gaussian_prune(step, where="pre_forward")

                # Keep opacity tensors in sync with densified gaussian count.
                self._sync_opacity_params()

                batch = self._move_batch_to_device(batch)

                render_output = self.model(
                    camera_to_world=batch["camera_to_world"],
                    intrinsics=batch["intrinsics"],
                    rgb_width=batch["rgb_width"],
                    rgb_height=batch["rgb_height"],
                    lidar_to_world=batch["lidar_to_world"],
                    lidar_width=batch["lidar_width"],
                    lidar_height=batch["lidar_height"],
                    lidar_vertical_fov_min_deg=self.config.dataset.lidar_vertical_fov_min_deg,
                    lidar_vertical_fov_max_deg=self.config.dataset.lidar_vertical_fov_max_deg,
                    lidar_vertical_angles_deg=self.config.dataset.lidar_vertical_angles_deg,
                    near_plane=self.config.dataset.near_plane,
                    far_plane=self.config.dataset.far_plane,
                )
                self._check_tensor_finite("render.rgb", render_output.rgb.rgb, step)
                self._check_tensor_finite("render.lidar.depth", render_output.lidar.depth, step)
                loss = self.criterion(batch, render_output, self.model)
                bad_terms = {lname: lval for lname, lval in loss.terms.items() if not np.isfinite(lval)}
                if bad_terms:
                    print(f"[nan-probe] step={step} loss_terms={bad_terms}", flush=True)
                    raise FloatingPointError(f"Non-finite loss terms at step={step}: {list(bad_terms)}")
                self._check_tensor_finite("loss.total", loss.total, step)
                debug_loss_terms = os.getenv("XSIM_LOSS_TERM_PROBE", "").strip().lower() in {"1", "true", "yes", "on"}
                probe_every = max(1, int(os.getenv("XSIM_LOSS_TERM_PROBE_EVERY", "200")))
                if debug_loss_terms and step % probe_every == 0:
                    if self._probe_loss_term_grads(step, loss.raw_terms or {}, batch, render_output):
                        continue

                # Densification/pruning (gsplat strategy)
                info = getattr(render_output, "rgb").extras if hasattr(render_output, "rgb") else {}
                if self.strategy is not None and self.strategy_state is not None:
                    # Opacities are stored as logits (conventional 3DGS); do not clamp.
                    self._sanitize_opacity_params("pre_strategy")

                    n_before = int(self.params["means"].shape[0])
                    debug_mcmc = (
                        isinstance(self.strategy, gsplat.MCMCStrategy)
                        and os.getenv("GSPLAT_MCMC_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
                    )
                    if debug_mcmc and self.densify_cfg is not None and int(self.densify_cfg.refine_every) > 0:
                        if step % int(self.densify_cfg.refine_every) == 0:
                            opa = torch.sigmoid(self.params["opacities"].detach()).flatten()
                            dead = opa <= float(getattr(self.densify_cfg, "prune_opa", 0.005))
                            print(
                                f"[mcmc-debug] pre step={step} n={n_before} "
                                f"opa_min={float(opa.min().item()):.6e} opa_max={float(opa.max().item()):.6e} "
                                f"opa_mean={float(opa.mean().item()):.6e} dead={int(dead.sum().item())}",
                                flush=True,
                            )
                    self.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state, step, info)

                    # IMPORTANT: gsplat strategies mutate/replace entries in `params`.
                    # Keep the nn.Module parameters in sync; otherwise the renderer keeps using
                    # stale tensors and N will appear unchanged.
                    for k in ("means", "sh_coeffs", "scales", "quats", "opacities"):
                        if k in self.params:
                            if k == "opacities" and self.model.use_separate_opacity:
                                # Strategy may replace opacities; keep camera opacity in sync.
                                if self.model.opacity_camera is not self.params["opacities"]:
                                    self.model.opacity_camera = self.params["opacities"]
                                    self.params["opacity_camera"] = self.params["opacities"]
                            elif hasattr(self.model, k):
                                if getattr(self.model, k) is not self.params[k]:
                                    setattr(self.model, k, self.params[k])

                    n_mid = int(self.params["means"].shape[0])
                    if (
                        self.densify_cfg is not None
                        and step >= int(self.densify_cfg.refine_start_iter)
                        and step <= int(self.densify_cfg.refine_stop_iter)
                        and int(self.densify_cfg.refine_every) > 0
                        and (step % int(self.densify_cfg.refine_every) == 0)
                    ):
                        pass

                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)

                loss.total.backward()
                self._sanitize_opacity_grads("post_backward")

                if self.strategy is not None and self.strategy_state is not None:
                    n_before = int(self.params["means"].shape[0])
                    if isinstance(self.strategy, gsplat.DefaultStrategy):
                        self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step, info, packed=True)
                    else:
                        # gsplat 1.5.3 MCMCStrategy expects an lr argument instead of `packed`.
                        # Use mean LR as the strategy noise step scale.
                        try:
                            self.strategy.step_post_backward(
                                self.params,
                                self.optimizers,
                                self.strategy_state,
                                step,
                                info,
                                lr=float(self.config.optimizer.lr_mean),
                            )
                        except Exception as exc:
                            if debug_mcmc:
                                opa = torch.sigmoid(self.params["opacities"].detach()).flatten()
                                print(
                                    f"[mcmc-debug] exception step={step} n={int(self.params['means'].shape[0])} "
                                    f"opa_min={float(opa.min().item()):.6e} opa_max={float(opa.max().item()):.6e} "
                                    f"opa_mean={float(opa.mean().item()):.6e} exc={type(exc).__name__}: {exc}",
                                    flush=True,
                                )
                            raise
                    self._sanitize_opacity_params("post_strategy")
                    self._sanitize_opacity_optimizer_state("post_strategy")
                    if debug_mcmc and self.densify_cfg is not None and int(self.densify_cfg.refine_every) > 0:
                        if step % int(self.densify_cfg.refine_every) == 0:
                            opa = torch.sigmoid(self.params["opacities"].detach()).flatten()
                            print(
                                f"[mcmc-debug] post step={step} n={int(self.params['means'].shape[0])} "
                                f"opa_min={float(opa.min().item()):.6e} opa_max={float(opa.max().item()):.6e} "
                                f"opa_mean={float(opa.mean().item()):.6e}",
                                flush=True,
                            )

                    # See note above: sync mutated params back onto the model.
                    for k in ("means", "sh_coeffs", "scales", "quats", "opacities"):
                        if k in self.params and hasattr(self.model, k):
                            if getattr(self.model, k) is not self.params[k]:
                                setattr(self.model, k, self.params[k])

                    n_after = int(self.params["means"].shape[0])
                self._sanitize_opacity_optimizer_state("pre_opt_step")
                for opt in self.optimizers.values():
                    opt.step()
                self._sanitize_opacity_params("post_opt_step")
                self._sanitize_opacity_optimizer_state("post_opt_step")

                # Opacities are stored as logits (conventional 3DGS); do not clamp.

                # Full-sequence evaluation every 200 steps on the test split.
                if step % 200 == 0:
                    try:
                        psnr_sum = 0.0
                        lidar_abs_sum = 0.0
                        lidar_count = 0
                        n_frames = 0
                        self.model.eval()
                        with torch.no_grad():
                            for ebatch in self.eval_loader:
                                self._sync_opacity_params()
                                ebatch = self._move_batch_to_device(ebatch)
                                eout = self.model(
                                    camera_to_world=ebatch["camera_to_world"],
                                    intrinsics=ebatch["intrinsics"],
                                    rgb_width=ebatch["rgb_width"],
                                    rgb_height=ebatch["rgb_height"],
                                    lidar_to_world=ebatch["lidar_to_world"],
                                    lidar_width=ebatch["lidar_width"],
                                    lidar_height=ebatch["lidar_height"],
                                    lidar_vertical_fov_min_deg=self.config.dataset.lidar_vertical_fov_min_deg,
                                    lidar_vertical_fov_max_deg=self.config.dataset.lidar_vertical_fov_max_deg,
                                    near_plane=self.config.dataset.near_plane,
                                    far_plane=self.config.dataset.far_plane,
                                    lidar_vertical_angles_deg=self.config.dataset.lidar_vertical_angles_deg,
                                )

                                pred_rgb = getattr(eout, "rgb").rgb.detach().float()
                                gt_rgb = ebatch["rgb"].detach().float()  # type: ignore[union-attr]
                                mse = (pred_rgb - gt_rgb).pow(2).mean().clamp_min(1.0e-12)
                                psnr_sum += float((-10.0 * torch.log10(mse)).item())

                                pred_lidar = getattr(eout, "lidar").depth.detach().float()
                                gt_lidar = ebatch["lidar_depth"].detach().float()  # type: ignore[union-attr]
                                valid = gt_lidar > 0
                                if bool(valid.any()):
                                    diff = (pred_lidar[valid] - gt_lidar[valid]).abs()
                                    lidar_abs_sum += float(diff.sum().item())
                                    lidar_count += int(valid.sum().item())
                                n_frames += 1
                        depth_l1 = lidar_abs_sum / max(lidar_count, 1)
                        avg_psnr = psnr_sum / max(n_frames, 1)
                        print(
                            f"[eval@{step:06d}] test_frames={n_frames} depth_l1={depth_l1:.4f} "
                            f"psnr={avg_psnr:.3f}",
                            flush=True,
                        )
                    except Exception as e:
                        print(f"[eval@{step:06d}] failed: {e}", flush=True)
                    finally:
                        self.model.train()

                if step % self.config.training.vis_every == 0:
                    self._save_visualization(step, batch, render_output)

                if step % self.config.training.save_every == 0:
                    self._save_checkpoint(step)
                    self._save_geometry(step, batch)

                # Keep aggregated optimizer state in sync for checkpoint saving.
                # Note: per-parameter optimizers are the source of truth. The aggregated optimizer
                # is only kept for backward compatibility with old checkpoints.

        self._save_checkpoint(step)

    def _move_batch_to_device(self, batch: dict[str, object]) -> dict[str, object]:
        moved: dict[str, object] = {}
        for key, value in batch.items():
            moved[key] = value.to(self.device) if isinstance(value, torch.Tensor) else value
        return moved

    @staticmethod
    def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
        # rgb: [3,H,W] float in [0,1]
        rgb = rgb.detach().float().clamp(0.0, 1.0).cpu()
        arr = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return arr

    @staticmethod
    def _project_lidar_to_rgb_overlay(
        rgb: torch.Tensor,
        lidar_points: torch.Tensor,
        camera_to_world: torch.Tensor,
        lidar_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        color: tuple[int, int, int] = (255, 0, 0),
    ) -> np.ndarray:
        """Project LiDAR points (LiDAR frame) onto an RGB image.

        Assumes dataset provides camera_to_world (T_wc), lidar_to_world (T_wl), and
        intrinsics K. Uses T_cl = inv(T_wc) @ T_wl.
        """
        # Base image
        img = JointTrainer._to_uint8_rgb(rgb).copy()  # [H,W,3]
        H, W = img.shape[0], img.shape[1]

        if lidar_points.ndim != 2 or lidar_points.shape[1] != 3:
            return img

        device = camera_to_world.device
        pts_l = lidar_points.to(device=device, dtype=torch.float32)
        if pts_l.numel() == 0:
            return img

        # Compute T_cl
        T_wc = camera_to_world.to(device=device, dtype=torch.float32)
        T_wl = lidar_to_world.to(device=device, dtype=torch.float32)
        T_cw = torch.linalg.inv(T_wc)
        T_cl = T_cw @ T_wl

        ones = torch.ones((pts_l.shape[0], 1), device=device, dtype=torch.float32)
        pts_l_h = torch.cat([pts_l, ones], dim=-1)  # [N,4]
        pts_c = (T_cl @ pts_l_h.T).T[:, :3]  # [N,3]

        # Project
        K = intrinsics.to(device=device, dtype=torch.float32)
        x, y, z = pts_c[:, 0], pts_c[:, 1], pts_c[:, 2]
        valid = z > 1.0e-6
        if not valid.any():
            return img

        x = x[valid]
        y = y[valid]
        z = z[valid]
        u = (K[0, 0] * (x / z) + K[0, 2]).round().to(torch.int64)
        v = (K[1, 1] * (y / z) + K[1, 2]).round().to(torch.int64)

        inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not inb.any():
            return img

        u = u[inb].cpu().numpy()
        v = v[inb].cpu().numpy()

        # Draw a small dilated halo so sparse LiDAR points are easier to see in RGB overlays.
        mask = np.zeros((H, W), dtype=bool)
        mask[v, u] = True
        halo = np.zeros_like(mask)
        radius = 2
        for dy in range(-radius, radius + 1):
            src_y0 = max(0, -dy)
            src_y1 = H - max(0, dy)
            dst_y0 = max(0, dy)
            dst_y1 = H - max(0, -dy)
            for dx in range(-radius, radius + 1):
                src_x0 = max(0, -dx)
                src_x1 = W - max(0, dx)
                dst_x0 = max(0, dx)
                dst_x1 = W - max(0, -dx)
                halo[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]

        r, g, b = color
        red = np.array([r, g, b], dtype=np.float32)
        img_f = img.astype(np.float32)
        img_f[halo] = 0.25 * img_f[halo] + 0.75 * red
        img = np.clip(img_f, 0.0, 255.0).round().astype(np.uint8)
        return img

    @staticmethod
    def _depth_to_color(depth: torch.Tensor, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
        # depth: [1,H,W] float, 0 means invalid
        d = depth.detach().float().cpu()[0]
        valid = d > 0
        if not valid.any():
            return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

        vals = d[valid]
        lo = float(vals.min()) if vmin is None else float(vmin)
        hi = float(vals.max()) if vmax is None else float(vmax)
        hi = max(hi, lo + 1.0e-6)

        x = (d - lo) / (hi - lo)
        x = x.clamp(0.0, 1.0)

        # Colorblind-friendly "viridis-like" gradient (dark purple -> yellow)
        r = 0.2803 + 0.7062 * x
        g = 0.1359 + 0.8536 * x
        b = 0.4899 - 0.4899 * x
        color = torch.stack([r, g, b], dim=-1).clamp(0.0, 1.0)
        color[~valid] = 0.0
        return (color.numpy() * 255.0).round().astype(np.uint8)

    def _save_visualization(self, step: int, batch: dict[str, object], render_output: object) -> None:
        # Local import to keep base deps minimal.
        from PIL import Image  # type: ignore

        pred_rgb = getattr(render_output, "rgb").rgb
        pred_cam_depth = getattr(render_output, "rgb").depth  # camera depth from rasterizer
        pred_lidar = getattr(render_output, "lidar").depth

        gt_rgb = batch["rgb"]
        gt_lidar = batch["lidar_depth"]

        out_dir = self.vis_dir / f"step_{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        Image.fromarray(self._to_uint8_rgb(gt_rgb)).save(out_dir / "gt_rgb.png")
        Image.fromarray(self._to_uint8_rgb(pred_rgb)).save(out_dir / "pred_rgb.png")

        # (C) Project raw LiDAR points to the GT RGB image for quick sanity check.
        try:
            if "lidar_points" in batch:
                overlay = self._project_lidar_to_rgb_overlay(
                    rgb=gt_rgb,
                    lidar_points=batch["lidar_points"],
                    camera_to_world=batch["camera_to_world"],
                    lidar_to_world=batch["lidar_to_world"],
                    intrinsics=batch["intrinsics"],
                )
                Image.fromarray(overlay).save(out_dir / "gt_rgb_lidar_overlay.png")
        except Exception:
            pass

        # RGB error heatmap (absolute error mean over channels)
        err = (pred_rgb.detach().float().cpu() - gt_rgb.detach().float().cpu()).abs().mean(dim=0, keepdim=True)
        err_color = self._depth_to_color(err, vmin=0.0, vmax=float(err.max().clamp_min(1.0e-6)))
        Image.fromarray(err_color).save(out_dir / "rgb_abs_err.png")

        # Depth visualizations
        # (A) Camera depth (from rasterizer). Use per-frame min/max on valid pixels.
        cam_d = pred_cam_depth.detach().float().cpu()[0]
        cam_valid = cam_d > 0
        if cam_valid.any():
            cam_vals = cam_d[cam_valid]
            cam_lo, cam_hi = float(cam_vals.min()), float(cam_vals.max())
        else:
            cam_lo, cam_hi = None, None
        Image.fromarray(self._depth_to_color(pred_cam_depth, vmin=cam_lo, vmax=cam_hi)).save(out_dir / "pred_cam_depth.png")
        np.save(out_dir / "pred_cam_depth.npy", pred_cam_depth.detach().float().cpu().numpy())

        # (B) LiDAR depth: use shared min/max from GT valid pixels for comparability
        gt_valid = gt_lidar.detach().float().cpu()[0] > 0
        if gt_valid.any():
            vals = gt_lidar.detach().float().cpu()[0][gt_valid]
            lo, hi = float(vals.min()), float(vals.max())
        else:
            lo, hi = None, None

        Image.fromarray(self._depth_to_color(gt_lidar, vmin=lo, vmax=hi)).save(out_dir / "gt_lidar_depth.png")
        pred_vals = pred_lidar.detach().float().cpu()[0]
        pred_valid = pred_vals > 0
        if bool(pred_valid.any()):
            lo_p = float(pred_vals[pred_valid].quantile(0.02))
            hi_p = float(pred_vals[pred_valid].quantile(0.98))
        else:
            lo_p, hi_p = 0.0, 1.0
        Image.fromarray(self._depth_to_color(pred_lidar, vmin=lo, vmax=hi)).save(out_dir / "pred_lidar_depth.png")
        Image.fromarray(self._depth_to_color(pred_lidar, vmin=lo_p, vmax=hi_p)).save(out_dir / "pred_lidar_depth_robust.png")

        # Save raw arrays for later analysis
        np.save(out_dir / "gt_lidar_depth.npy", gt_lidar.detach().float().cpu().numpy())
        np.save(out_dir / "pred_lidar_depth.npy", pred_lidar.detach().float().cpu().numpy())

    def _save_checkpoint(self, step: int) -> None:
        path = self.checkpoint_dir / f"step_{step:06d}.pt"
        # Optimizer states can become inconsistent if parameters are replaced (densify or opacity sync).
        # Guard against state_dict errors to keep checkpoints flowing.
        opt_states: dict[str, dict] = {}
        for k, opt in self.optimizers.items():
            try:
                opt_states[k] = opt.state_dict()
            except Exception as e:
                pass
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optimizers": opt_states,
                "strategy": None if self.strategy is None else {
                    "type": type(self.strategy).__name__,
                    "state": self.strategy_state,
                },
                "config": self.config,
            },
            path,
        )

    def _export_gsplat_ply(self, path: Path) -> None:
        """Export gaussians in an Inria-3DGS compatible *binary* PLY.

        SuperSplat and many third-party viewers expect the same schema/order as the
        original gaussian-splatting repo's point_cloud.ply:
          x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
        with `format binary_little_endian 1.0`.
        """
        import struct

        means = self.model.means.detach().float().cpu().numpy().astype(np.float32)  # [N,3]
        N = int(means.shape[0])
        if N == 0:
            return

        # Opacity in [0,1]
        if hasattr(self.model, "get_camera_opacity"):
            opacities = self.model.get_camera_opacity().detach().float().cpu().numpy().reshape(N, 1).astype(np.float32)
        elif hasattr(self.model, "opacities"):
            opacities = self.model.opacities.detach().float().cpu().numpy().reshape(N, 1).astype(np.float32)
        else:
            opacities = np.ones((N, 1), dtype=np.float32)

        # Scales: export log-scales (Inria schema expects log-scale values)
        log_scales = self.model.scales.detach().float().cpu().numpy().astype(np.float32)  # [N,3]

        # Rotations: export normalized quaternion as (w,x,y,z)
        quats = self.model.quats.detach().float().cpu().numpy().astype(np.float32)  # [N,4]
        qnorm = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.clip(qnorm, 1.0e-8, None)
        # internal is (x,y,z,w) -> export (w,x,y,z)
        rots = np.concatenate([quats[:, 3:4], quats[:, 0:3]], axis=1).astype(np.float32)  # [N,4]

        # Normals: not used by splatting; set to zero
        normals = np.zeros((N, 3), dtype=np.float32)

        # SH coeffs: degree-3 -> 16 bases * 3 = 48 floats
        sh = self.model.sh_coeffs.detach().float().cpu().numpy().astype(np.float32)  # [N,16,3]
        sh = sh.reshape(N, -1)

        # Defensive: strategy may have changed N between reading means and sh.
        N = min(N, sh.shape[0], opacities.shape[0], log_scales.shape[0], rots.shape[0])
        means = means[:N]
        opacities = opacities[:N]
        log_scales = log_scales[:N]
        rots = rots[:N]
        sh = sh[:N]
        f_dc = sh[:, 0:3].astype(np.float32)     # [N,3]
        f_rest = sh[:, 3:].astype(np.float32)   # [N,45]
        if f_rest.shape[1] != 45:
            raise ValueError(f"Expected 45 f_rest coeffs for SH degree 3, got {f_rest.shape[1]}")

        # Write binary little endian PLY
        with path.open("wb") as f:
            header = []
            header.append("ply")
            header.append("format binary_little_endian 1.0")
            header.append(f"element vertex {N}")
            header.append("property float x")
            header.append("property float y")
            header.append("property float z")
            header.append("property float nx")
            header.append("property float ny")
            header.append("property float nz")
            header.append("property float f_dc_0")
            header.append("property float f_dc_1")
            header.append("property float f_dc_2")
            for i in range(45):
                header.append(f"property float f_rest_{i}")
            header.append("property float opacity")
            header.append("property float scale_0")
            header.append("property float scale_1")
            header.append("property float scale_2")
            header.append("property float rot_0")
            header.append("property float rot_1")
            header.append("property float rot_2")
            header.append("property float rot_3")
            header.append("end_header")
            f.write(("\n".join(header) + "\n").encode("ascii"))

            pack = struct.Struct("<" + "f" * (3 + 3 + 3 + 45 + 1 + 3 + 4)).pack
            for i in range(N):
                row = (
                    float(means[i, 0]), float(means[i, 1]), float(means[i, 2]),
                    float(normals[i, 0]), float(normals[i, 1]), float(normals[i, 2]),
                    float(f_dc[i, 0]), float(f_dc[i, 1]), float(f_dc[i, 2]),
                    *[float(v) for v in f_rest[i]],
                    float(opacities[i, 0]),
                    float(log_scales[i, 0]), float(log_scales[i, 1]), float(log_scales[i, 2]),
                    float(rots[i, 0]), float(rots[i, 1]), float(rots[i, 2]), float(rots[i, 3]),
                )
                f.write(pack(*row))

        pass

    def _save_geometry(self, step: int, batch: dict[str, object]) -> None:
        """Save Gaussians + current frame camera/LiDAR poses for visualization."""
        out_dir = self.checkpoint_dir / "geometry"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Also export a standard 3D Gaussian Splatting PLY (xyz + SH + opacity + scale + rotation)
        # so you can load it in third-party viewers/renderers.
        try:
            self._export_gsplat_ply(out_dir / f"gaussians_gsplat_step_{step:06d}.ply")
        except Exception:
            pass

        means = self.model.means.detach().float().cpu().numpy()  # [N,3]
        if means.shape[0] == 0:
            return

        # Color: use SH DC if available, otherwise white.
        try:
            sh0 = self.model.sh_coeffs.detach().float().cpu().numpy()[:, 0, :]  # [N,3]
            colors = 1.0 / (1.0 + np.exp(-sh0))  # sigmoid
        except Exception:
            colors = np.ones_like(means, dtype=np.float32)
        colors_u8 = (np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Write ASCII PLY (no extra deps).
        ply_path = out_dir / f"gaussians_step_{step:06d}.ply"
        with ply_path.open("w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {means.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for (x, y, z), (r, g, b) in zip(means, colors_u8, strict=False):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")

        # Save current frame poses as simple line sets (PLY) and raw matrices (NPZ).
        cam_to_world = batch["camera_to_world"].detach().float().cpu().numpy()  # type: ignore[union-attr]
        lidar_to_world = batch["lidar_to_world"].detach().float().cpu().numpy()  # type: ignore[union-attr]

        npz_path = out_dir / f"poses_step_{step:06d}.npz"
        np.savez(npz_path, camera_to_world=cam_to_world, lidar_to_world=lidar_to_world)

        def _write_axes_ply(path, T, axis_len=1.0):
            R = T[:3, :3]
            t = T[:3, 3]
            origin = t
            x = t + R @ np.array([axis_len, 0.0, 0.0], dtype=np.float32)
            y = t + R @ np.array([0.0, axis_len, 0.0], dtype=np.float32)
            z = t + R @ np.array([0.0, 0.0, axis_len], dtype=np.float32)
            verts = np.stack([origin, x, y, z], axis=0)
            cols = np.array([
                [255, 255, 255],  # origin
                [255, 0, 0],      # x
                [0, 255, 0],      # y
                [0, 0, 255],      # z
            ], dtype=np.uint8)
            edges = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
            with open(path, "w", encoding="utf-8") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write(f"element edge {len(edges)}\n")
                f.write("property int vertex1\nproperty int vertex2\n")
                f.write("end_header\n")
                for (xv, yv, zv), (r, g, b) in zip(verts, cols, strict=False):
                    f.write(f"{xv:.6f} {yv:.6f} {zv:.6f} {int(r)} {int(g)} {int(b)}\n")
                for a, b in edges:
                    f.write(f"{int(a)} {int(b)}\n")

        _write_axes_ply(out_dir / f"camera_axes_step_{step:06d}.ply", cam_to_world, axis_len=1.0)
        _write_axes_ply(out_dir / f"lidar_axes_step_{step:06d}.ply", lidar_to_world, axis_len=1.0)

        # Save full trajectories for the sequence (if dataset exposes poses like KittiRDataset).
        if hasattr(self.dataset, "_lidar_poses") and hasattr(self.dataset, "_T_cam_velo"):
            lidar_poses = getattr(self.dataset, "_lidar_poses")
            T_cl = getattr(self.dataset, "_T_cam_velo")

            # lidar_poses is list[np.ndarray] or similar
            P_l = np.stack([np.asarray(p, dtype=np.float32)[:3, 3] for p in lidar_poses], axis=0)  # [F,3]

            # camera positions from Twc = Twl @ inv(Tcl)
            Tcl = np.asarray(T_cl, dtype=np.float32)
            inv_Tcl = np.linalg.inv(Tcl)
            P_c = []
            for Twl in lidar_poses:
                Twc = np.asarray(Twl, dtype=np.float32) @ inv_Tcl
                P_c.append(Twc[:3, 3])
            P_c = np.stack(P_c, axis=0).astype(np.float32)

            np.savez(
                out_dir / f"trajectories_step_{step:06d}.npz",
                lidar_positions=P_l,
                camera_positions=P_c,
            )

            def _write_polyline_ply(path, pts, color):
                pts = np.asarray(pts, dtype=np.float32)
                n = int(pts.shape[0])
                if n < 2:
                    return
                edges = np.stack([np.arange(n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
                with open(path, "w", encoding="utf-8") as f:
                    f.write("ply\nformat ascii 1.0\n")
                    f.write(f"element vertex {n}\n")
                    f.write("property float x\nproperty float y\nproperty float z\n")
                    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                    f.write(f"element edge {len(edges)}\n")
                    f.write("property int vertex1\nproperty int vertex2\n")
                    f.write("end_header\n")
                    r, g, b = color
                    for x, y, z in pts:
                        f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
                    for a, b2 in edges:
                        f.write(f"{int(a)} {int(b2)}\n")

            _write_polyline_ply(out_dir / f"lidar_trajectory_step_{step:06d}.ply", P_l, (255, 255, 0))
            _write_polyline_ply(out_dir / f"camera_trajectory_step_{step:06d}.ply", P_c, (0, 255, 255))

            # Save per-frame camera forward vectors along the trajectory as a single line-set PLY.
            # We export BOTH +Z and -Z directions (different colors) so you can quickly verify
            # which convention matches the actual viewing direction.
            vec_len = 1.5
            verts = []
            cols = []
            edges = []
            base = 0
            for Twl in lidar_poses:
                Twc = np.asarray(Twl, dtype=np.float32) @ inv_Tcl
                R = Twc[:3, :3]
                t = Twc[:3, 3]

                # +Z (cyan) and -Z (magenta) in camera coordinates
                o = t
                fwd_p = t + R @ np.array([0.0, 0.0, vec_len], dtype=np.float32)
                fwd_n = t + R @ np.array([0.0, 0.0, -vec_len], dtype=np.float32)

                verts.extend([o, fwd_p, o, fwd_n])
                cols.extend([
                    [255, 255, 255],  # origin for +Z
                    [0, 255, 255],    # +Z tip
                    [255, 255, 255],  # origin for -Z
                    [255, 0, 255],    # -Z tip
                ])
                edges.extend([[base + 0, base + 1], [base + 2, base + 3]])
                base += 4

            vec_path = out_dir / f"camera_forward_vectors_step_{step:06d}.ply"
            with open(vec_path, "w", encoding="utf-8") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {len(verts)}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write(f"element edge {len(edges)}\n")
                f.write("property int vertex1\nproperty int vertex2\n")
                f.write("end_header\n")
                for (xv, yv, zv), (r, g, b) in zip(verts, cols, strict=False):
                    f.write(f"{float(xv):.6f} {float(yv):.6f} {float(zv):.6f} {int(r)} {int(g)} {int(b)}\n")
                for a, b2 in edges:
                    f.write(f"{int(a)} {int(b2)}\n")
