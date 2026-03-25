from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

try:
    from gsplat.rendering import rasterization, rasterization_2dgs, rasterize_to_pixels, isect_tiles, isect_offset_encode
    from gsplat.cuda._wrapper import quat_scale_to_covar_preci
except Exception:  # pragma: no cover
    GSPLAT_IMPORT_ERROR = None
    rasterization = None
    rasterization_2dgs = None
    rasterize_to_pixels = None
    isect_tiles = None
    isect_offset_encode = None
    quat_scale_to_covar_preci = None

from .lidar_projection import elevation_to_row
from .lidar_models import build_gsplat_lidar_coeffs
from .lidar_resample import warp_depth_to_vertical_angles

@dataclass
class SensorRenderOutput:
    rgb: torch.Tensor
    depth: torch.Tensor
    alpha: torch.Tensor
    extras: dict[str, Any]


@dataclass
class JointRenderOutput:
    rgb: SensorRenderOutput
    lidar: SensorRenderOutput


class GaussianSceneModel(nn.Module):
    """3D Gaussian Scene Model (camera + LiDAR).

    RGB rendering uses gsplat's rasterizer.

    LiDAR rendering uses a spherical projection on the range image, with
    optional UT-based footprint estimation to better respect scale and rotation.
    """
    def __init__(
        self,
        num_gaussians: int,
        sh_degree: int,
        init_extent: float,
        init_opacity: float,
        init_scale: float,
        background_color: tuple[float, float, float],
        use_separate_opacity:bool=True,
        lidar_ut_enable: bool = False,
        lidar_ut_alpha: float = 1.0e-3,
        lidar_ut_beta: float = 2.0,
        lidar_ut_kappa: float = 0.0,
        lidar_ut_delta: float = -1.0,
        lidar_ut_in_image_margin_factor: float = 0.1,
        lidar_ut_require_all_sigma_points_valid: bool = False,
    ) -> None:
        super().__init__()
        self.sh_degree = sh_degree

        self.use_separate_opacity = use_separate_opacity
        self.register_buffer("background_color", torch.tensor(background_color, dtype=torch.float32), persistent=False)
        self._lidar_ut_enable = bool(lidar_ut_enable)
        self._lidar_ut_alpha = float(lidar_ut_alpha)
        self._lidar_ut_beta = float(lidar_ut_beta)
        self._lidar_ut_kappa = float(lidar_ut_kappa)
        self._lidar_ut_delta = float(lidar_ut_delta)
        self._lidar_ut_in_image_margin_factor = float(lidar_ut_in_image_margin_factor)
        self._lidar_ut_require_all_sigma_points_valid = bool(lidar_ut_require_all_sigma_points_valid)
        self._lidar_depth_aggregation = "mean"
        self._lidar_median_threshold = 0.5
        self._lidar_render_backend = "custom"
        self._lidar_render_config: Any | None = None
        self._lidar_coeffs_cache: dict[tuple[int, int, str, int], Any] = {}
        self._lidar_spherical_splat_mode = "bilinear"
        self._lidar_spherical_kernel_radius = 1
        self.means = nn.Parameter((torch.rand(num_gaussians, 3) - 0.5) * 2.0 * init_extent)
        # Conventional 3DGS parameterization: store *log-scales* as the learnable parameter.
        # IMPORTANT: keep the attribute name `scales` so gsplat strategies (and trainer sync)
        # can mutate/swap this tensor alongside means/quats/opacities.
        self.scales = nn.Parameter(torch.full((num_gaussians, 3), float(math.log(max(float(init_scale), 1.0e-6)))))
        self.quats = nn.Parameter(self._random_quaternions(num_gaussians))
        # Appearance parameters
        sh_bases = (sh_degree + 1) ** 2
        self.sh_coeffs = nn.Parameter(torch.randn(num_gaussians, sh_bases, 3) * 0.01)
        # Extended opacity representation (XSIM contribution)
        # Conventional 3DGS uses opacity logits, rendered with sigmoid.
        if use_separate_opacity:
            self.opacity_camera = nn.Parameter(torch.full((num_gaussians, 1), float(self._inverse_sigmoid(float(init_opacity)))))
            self.opacity_lidar = nn.Parameter(torch.full((num_gaussians, 1), float(self._inverse_sigmoid(float(init_opacity)))))
        else:
            # Single opacity logits (preferred for gsplat strategies)
            self.opacities = nn.Parameter(torch.full((num_gaussians, 1), float(self._inverse_sigmoid(float(init_opacity)))))

    def forward(
        self,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        rgb_width: int,
        rgb_height: int,
        lidar_to_world: torch.Tensor,
        lidar_width: int,
        lidar_height: int,
        lidar_vertical_fov_min_deg: float,
        lidar_vertical_fov_max_deg: float,
        near_plane: float,
        far_plane: float,
        lidar_vertical_angles_deg: list[float] | None = None,
    ) -> JointRenderOutput:
        rgb_render = self.render_rgb(
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            width=rgb_width,
            height=rgb_height,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        lidar_render = self.render_lidar(
            lidar_to_world=lidar_to_world,
            width=lidar_width,
            height=lidar_height,
            vertical_fov_min_deg=lidar_vertical_fov_min_deg,
            vertical_fov_max_deg=lidar_vertical_fov_max_deg,
            vertical_angles_deg=lidar_vertical_angles_deg,
            near_plane=near_plane,
            far_plane=far_plane,
            lidar_vertical_angles_deg=lidar_vertical_angles_deg,
        )
        return JointRenderOutput(rgb=rgb_render, lidar=lidar_render)

    def configure_lidar_model(self, config: Any) -> None:
        self._lidar_depth_aggregation = str(getattr(config, "lidar_depth_aggregation", "mean")).strip().lower()
        self._lidar_median_threshold = float(getattr(config, "lidar_median_threshold", 0.5))
        self._lidar_render_backend = str(getattr(config, "lidar_render_backend", "custom")).strip().lower()
        self._lidar_render_config = config
        self._lidar_coeffs_cache.clear()

    @property
    def log_scales(self) -> torch.Tensor:
        # Back-compat for code that expects `model.log_scales`.
        # Note: `self.scales` stores log-scales.
        return self.scales

    @staticmethod
    def _call_rasterizer(call_kwargs: dict[str, Any]) -> Any:
        if rasterization is None:
            raise ImportError("gsplat is required for RGB rendering.")
        return rasterization(**call_kwargs)

    def get_camera_opacity(self) -> torch.Tensor:
        """Get opacity values for camera rendering (σc), in [0,1], shape [N]."""
        if self.use_separate_opacity:
            logits = torch.nan_to_num(self.opacity_camera, nan=0.0, posinf=20.0, neginf=-20.0)
            opa = torch.sigmoid(logits).squeeze(-1)
        else:
            logits = torch.nan_to_num(self.opacities, nan=0.0, posinf=20.0, neginf=-20.0)
            opa = torch.sigmoid(logits).squeeze(-1)
        # Be tolerant to accidental leading singleton batch dims.
        if opa.ndim == 2 and opa.shape[0] == 1:
            opa = opa[0]
        return opa

    def get_lidar_opacity(self) -> torch.Tensor:
        """Get opacity values for LiDAR rendering (σL), in [0,1], shape [N]."""
        if self.use_separate_opacity:
            logits = torch.nan_to_num(self.opacity_lidar, nan=0.0, posinf=20.0, neginf=-20.0)
            opa = torch.sigmoid(logits).squeeze(-1)
        else:
            logits = torch.nan_to_num(self.opacities, nan=0.0, posinf=20.0, neginf=-20.0)
            opa = torch.sigmoid(logits).squeeze(-1)
        if opa.ndim == 2 and opa.shape[0] == 1:
            opa = opa[0]
        return opa

    def render_rgb(
        self,
        camera_to_world: torch.Tensor,
        intrinsics: torch.Tensor,
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        
    ) -> SensorRenderOutput:
        if rasterization is None:
            raise ImportError("gsplat is required for RGB rendering.") from GSPLAT_IMPORT_ERROR

        view_matrix = torch.linalg.inv(camera_to_world)
        world_to_camera = view_matrix

        means = self.means
        quats = self.quats
        log_scales = self.scales
        colors = torch.sigmoid(self.sh_coeffs[:, 0, :])

        # gsplat expects packed inputs with shape [N,3] (no batch dim) in our use.
        # Some strategy updates may introduce a leading singleton batch dim; strip it.
        if means.ndim == 3:
            # Flatten batch dims to keep packed [N,3] convention for gsplat.
            B, N = int(means.shape[0]), int(means.shape[1])
            means = means.reshape(B * N, 3)
            if quats.ndim == 3:
                quats = quats.reshape(B * N, 4)
            if log_scales.ndim == 3:
                log_scales = log_scales.reshape(B * N, 3)
            if colors.ndim == 3:
                colors = colors.reshape(B * N, colors.shape[-1])

        opacities = self.get_camera_opacity()
        if means.ndim == 2:
            # Align opacities to flattened means if a batch dim was collapsed.
            if opacities.ndim == 2:
                opacities = opacities.reshape(-1)
            elif opacities.ndim == 1 and opacities.shape[0] != means.shape[0]:
                # Pad or repeat to match means count after densify.
                if means.shape[0] > opacities.shape[0]:
                    pad = means.shape[0] - opacities.shape[0]
                    opacities = torch.cat([opacities, opacities[-1:].repeat(pad)], dim=0)
                else:
                    opacities = opacities[:means.shape[0]]

        call_kwargs = {
            "means": means,
            "quats": F.normalize(quats, dim=-1),
            "scales": torch.exp(log_scales).clamp_min(1.0e-6),
            "opacities": opacities,  # [N] or [1, N] if batched
            "colors": colors,
            "viewmats": world_to_camera.unsqueeze(0),
            "Ks": intrinsics.unsqueeze(0),
            "width": width,
            "height": height,
            "near_plane": near_plane,
            "far_plane": far_plane,
            "backgrounds": self.background_color,
            "sh_degree": None,
            # Needed by gsplat densification/pruning strategies (DefaultStrategy/MCMCStrategy)
            "absgrad": getattr(self, "_strategy_absgrad", False),
            "packed": True,
        }
        render_result = self._call_rasterizer(call_kwargs)
        rgb, depth, alpha, extras = self._normalize_render_result(render_result, height, width)
        return SensorRenderOutput(rgb=rgb, depth=depth, alpha=alpha, extras=extras)
    def render_lidar(
        self,
        lidar_to_world: torch.Tensor,
        width: int,
        height: int,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        near_plane: float,
        far_plane: float,
        vertical_angles_deg: list[float] | None = None,
        lidar_vertical_angles_deg: list[float] | None = None,
    ) -> SensorRenderOutput:
        world_to_lidar = torch.linalg.inv(lidar_to_world)
        means = self.means
        if means.ndim == 3 and means.shape[0] == 1:
            means = means[0]

        means_l = self._transform_points(world_to_lidar, means)
        scales = torch.exp(self.scales).clamp_min(1.0e-6)
        opacity = self.get_lidar_opacity()
        target_vertical_angles_deg = vertical_angles_deg or lidar_vertical_angles_deg
        if self._lidar_render_backend == "gsplat_ut":
            depth_map, alpha_map = self._render_lidar_gsplat_ut(
                means=means,
                scales=scales,
                opacity=opacity,
                width=width,
                height=height,
                lidar_to_world=lidar_to_world,
                near_plane=near_plane,
                far_plane=far_plane,
            )
            rgb_map = torch.zeros((3, height, width), device=depth_map.device, dtype=depth_map.dtype)
            return SensorRenderOutput(
                rgb=rgb_map,
                depth=depth_map,
                alpha=alpha_map,
                extras={
                    "lidar_impl": "gsplat_ut",
                    "lidar_sampling": "native_ut",
                },
            )

        if self._lidar_render_backend == "spherical_proxy_ut":
            depth_map, alpha_map = self._render_lidar_spherical_proxy_ut(
                means_l=means_l,
                scales=scales,
                opacity=opacity,
                lidar_to_world=lidar_to_world,
                width=width,
                height=height,
                vertical_fov_min_deg=vertical_fov_min_deg,
                vertical_fov_max_deg=vertical_fov_max_deg,
                near_plane=near_plane,
                far_plane=far_plane,
            )
            if target_vertical_angles_deg is not None and len(target_vertical_angles_deg) > 0:
                warped_alpha = warp_depth_to_vertical_angles(
                    depth=alpha_map,
                    target_vertical_angles_deg=target_vertical_angles_deg,
                    source_vertical_fov_min_deg=vertical_fov_min_deg,
                    source_vertical_fov_max_deg=vertical_fov_max_deg,
                )
                warped_depth = warp_depth_to_vertical_angles(
                    depth=depth_map * alpha_map,
                    target_vertical_angles_deg=target_vertical_angles_deg,
                    source_vertical_fov_min_deg=vertical_fov_min_deg,
                    source_vertical_fov_max_deg=vertical_fov_max_deg,
                )
                depth_map = warped_depth / warped_alpha.clamp_min(1.0e-10)
                depth_map = depth_map * (warped_alpha > (1.0 / 255.0)).to(depth_map.dtype)
                alpha_map = warped_alpha
            rgb_map = torch.zeros((3, height, width), device=depth_map.device, dtype=depth_map.dtype)
            return SensorRenderOutput(
                rgb=rgb_map,
                depth=depth_map,
                alpha=alpha_map,
                extras={
                    "lidar_impl": "spherical_proxy_ut",
                    "lidar_sampling": "uniform_then_angle_resample" if target_vertical_angles_deg else "projection_only",
                },
            )

        if self._lidar_render_backend in {"gsplat_ut_proj", "spherical_proxy"}:
            depth_map, alpha_map = self._render_lidar_spherical_proxy(
                means_l=means_l,
                scales=scales,
                opacity=opacity,
                lidar_to_world=lidar_to_world,
                width=width,
                height=height,
                vertical_fov_min_deg=vertical_fov_min_deg,
                vertical_fov_max_deg=vertical_fov_max_deg,
                near_plane=near_plane,
                far_plane=far_plane,
            )
            if target_vertical_angles_deg is not None and len(target_vertical_angles_deg) > 0:
                warped_alpha = warp_depth_to_vertical_angles(
                    depth=alpha_map,
                    target_vertical_angles_deg=target_vertical_angles_deg,
                    source_vertical_fov_min_deg=vertical_fov_min_deg,
                    source_vertical_fov_max_deg=vertical_fov_max_deg,
                )
                warped_depth = warp_depth_to_vertical_angles(
                    depth=depth_map * alpha_map,
                    target_vertical_angles_deg=target_vertical_angles_deg,
                    source_vertical_fov_min_deg=vertical_fov_min_deg,
                    source_vertical_fov_max_deg=vertical_fov_max_deg,
                )
                depth_map = warped_depth / warped_alpha.clamp_min(1.0e-10)
                depth_map = depth_map * (warped_alpha > (1.0 / 255.0)).to(depth_map.dtype)
                alpha_map = warped_alpha
            rgb_map = torch.zeros((3, height, width), device=depth_map.device, dtype=depth_map.dtype)
            return SensorRenderOutput(
                rgb=rgb_map,
                depth=depth_map,
                alpha=alpha_map,
                extras={
                    "lidar_impl": "spherical_proxy",
                    "lidar_sampling": "uniform_then_angle_resample" if target_vertical_angles_deg else "projection_only",
                },
            )

        depth_map, alpha_map = self._render_lidar_gsplat_spherical(
            means_l=means_l,
            scales=scales,
            opacity=opacity,
            width=width,
            height=height,
            vertical_fov_min_deg=vertical_fov_min_deg,
            vertical_fov_max_deg=vertical_fov_max_deg,
            vertical_angles_deg=None,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        if target_vertical_angles_deg is not None and len(target_vertical_angles_deg) > 0:
            warped_alpha = warp_depth_to_vertical_angles(
                depth=alpha_map,
                target_vertical_angles_deg=target_vertical_angles_deg,
                source_vertical_fov_min_deg=vertical_fov_min_deg,
                source_vertical_fov_max_deg=vertical_fov_max_deg,
            )
            warped_depth = warp_depth_to_vertical_angles(
                depth=depth_map * alpha_map,
                target_vertical_angles_deg=target_vertical_angles_deg,
                source_vertical_fov_min_deg=vertical_fov_min_deg,
                source_vertical_fov_max_deg=vertical_fov_max_deg,
            )
            depth_map = warped_depth / warped_alpha.clamp_min(1.0e-10)
            depth_map = depth_map * (warped_alpha > (1.0 / 255.0)).to(depth_map.dtype)
            alpha_map = warped_alpha
        rgb_map = torch.zeros((3, height, width), device=depth_map.device, dtype=depth_map.dtype)
        return SensorRenderOutput(
            rgb=rgb_map,
            depth=depth_map,
            alpha=alpha_map,
            extras={
                "lidar_impl": "gsplat_spherical",
                "lidar_sampling": "uniform_then_angle_resample" if target_vertical_angles_deg else "uniform",
            },
        )

    def _get_lidar_coeffs(self, width: int, height: int, device: torch.device) -> Any:
        if self._lidar_render_config is None:
            raise RuntimeError("Lidar render config is not set. Call configure_lidar_model() first.")
        key = (int(width), int(height), device.type, int(device.index if device.index is not None else -1))
        cached = self._lidar_coeffs_cache.get(key)
        if cached is not None:
            return cached
        coeffs = build_gsplat_lidar_coeffs(self._lidar_render_config, width=width, height=height, device=device)
        self._lidar_coeffs_cache[key] = coeffs
        return coeffs

    def _render_lidar_spherical_proxy(
        self,
        means_l: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        lidar_to_world: torch.Tensor,
        width: int,
        height: int,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        near_plane: float,
        far_plane: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rasterize_to_pixels is None or isect_tiles is None or isect_offset_encode is None:
            raise ImportError("gsplat rasterization helpers are required for the spherical proxy backend.")
        if means_l.ndim != 2 or means_l.shape[-1] != 3:
            raise ValueError(f"means_l must be [N,3], got {tuple(means_l.shape)}")

        device = means_l.device
        dtype = means_l.dtype
        eps = 1.0e-6
        min_alpha = 1.0 / 255.0
        eps2d = 0.3

        x = means_l[:, 0]
        y = means_l[:, 1]
        z = means_l[:, 2]
        xy2 = (x.square() + y.square()).clamp_min(eps)
        xy = torch.sqrt(xy2)
        r2 = (xy2 + z.square()).clamp_min(eps)
        r = torch.sqrt(r2)
        az = torch.atan2(y, x)
        el = torch.atan2(z, xy)

        min_el = math.radians(float(vertical_fov_min_deg))
        max_el = math.radians(float(vertical_fov_max_deg))
        span = max(max_el - min_el, eps)
        valid = (r >= float(near_plane)) & (r <= float(far_plane)) & (el >= min_el) & (el <= max_el)

        u = (az + math.pi) / (2.0 * math.pi) * float(width)
        v = (max_el - el) / span * float(max(height - 1, 1))
        means2d = torch.stack([u, v], dim=-1).unsqueeze(0)
        # Virtual 3DGS center in the LiDAR-proxy camera coordinate frame.
        # The mean/median branches use the same proxy geometry; only the
        # rasterization backend differs.
        denom_xy2 = (x.square() + y.square()).clamp_min(eps)
        fx = (float(width) / (2.0 * math.pi)) * (r / denom_xy2.sqrt())
        fy = (float(height) / span) * (r / (r + 1.0e-6))
        fx = fx.clamp(min=1.0, max=float(width) * 4.0)
        fy = fy.clamp(min=1.0, max=float(height) * 4.0)
        fx_g = float(torch.median(fx).item())
        fy_g = float(torch.median(fy).item())
        K_global = torch.tensor(
            [[fx_g, 0.0, float(width) * 0.5], [0.0, fy_g, float(height) * 0.5], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        Xn = (u - K_global[0, 2]) / K_global[0, 0]
        Yn = (v - K_global[1, 2]) / K_global[1, 1]
        means_syn = torch.stack([Xn * r, Yn * r, r], dim=-1)

        cov_w, _ = quat_scale_to_covar_preci(F.normalize(self.quats, dim=-1), scales, compute_covar=True, compute_preci=False)
        if cov_w is None:
            raise RuntimeError("Failed to compute covariance from quaternions and scales.")
        if cov_w.ndim != 3 or cov_w.shape[-2:] != (3, 3):
            raise ValueError(f"Unexpected covariance shape: {tuple(cov_w.shape)}")

        world_to_lidar = torch.linalg.inv(lidar_to_world)
        R = world_to_lidar[:3, :3]
        cov_l = torch.einsum("ij,njk,kl->nil", R, cov_w, R.transpose(0, 1)).unsqueeze(0)

        du_dx = -(float(width) / (2.0 * math.pi)) * (y / xy2)
        du_dy = (float(width) / (2.0 * math.pi)) * (x / xy2)
        du_dz = torch.zeros_like(du_dx)

        k = float(max(height - 1, 1)) / span
        dv_dx = k * (z * x) / (xy.clamp_min(eps) * r2)
        dv_dy = k * (z * y) / (xy.clamp_min(eps) * r2)
        dv_dz = -k * (xy / r2)

        J = torch.stack(
            [
                torch.stack([du_dx, du_dy, du_dz], dim=-1),
                torch.stack([dv_dx, dv_dy, dv_dz], dim=-1),
            ],
            dim=-2,
        ).unsqueeze(0)  # [1, N, 2, 3]
        cov2d = torch.einsum("bnij,bnjk,bnlk->bnil", J, cov_l, J)
        cov2d = 0.5 * (cov2d + cov2d.transpose(-1, -2))

        det_orig = torch.linalg.det(cov2d)
        cov2d = cov2d + eps2d * torch.eye(2, device=device, dtype=dtype)
        det_blur = torch.linalg.det(cov2d)
        compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0.0))

        cov2d_inv = torch.linalg.inv(cov2d)

        cov_diag = torch.diagonal(cov2d, dim1=-2, dim2=-1)
        trace = cov_diag.sum(dim=-1)
        b = 0.5 * trace
        tmp = torch.sqrt(torch.clamp(b * b - det_blur, min=0.01))
        v1 = b + tmp

        extend = torch.full_like(r, 3.33)
        if opacity is not None:
            eff_opacity = opacity * compensation
            extend = torch.minimum(
                extend,
                torch.sqrt(
                    torch.clamp(2.0 * torch.log(torch.clamp(eff_opacity / min_alpha, min=1.0)), min=0.0)
                ),
            )
        radius = torch.ceil(
            torch.minimum(
                extend[..., None] * torch.sqrt(cov_diag.clamp_min(eps)),
                (extend * torch.sqrt(v1))[..., None],
            )
        )
        radii = radius.to(torch.int32)

        valid = valid & (det_blur > 0.0) & (torch.max(radii, dim=-1)[0] > 0)
        means2d = torch.where(valid[..., None], means2d, torch.zeros_like(means2d))
        depths = torch.where(valid, r, torch.zeros_like(r))
        conics = torch.stack(
            [cov2d_inv[..., 0, 0], cov2d_inv[..., 0, 1], cov2d_inv[..., 1, 1]],
            dim=-1,
        )
        conics = torch.where(valid[..., None], conics, torch.zeros_like(conics))
        radii = torch.where(valid[..., None], radii, torch.zeros_like(radii))
        eff_opacity = torch.where(valid, opacity * compensation, torch.zeros_like(opacity))
        colors = depths[..., None]
        view = torch.eye(4, device=device, dtype=dtype)

        depth_agg = self._lidar_depth_aggregation
        if depth_agg == "median":
            if rasterization_2dgs is None:
                raise ImportError("gsplat 2DGS rasterization is required for LiDAR median depth.")
            means_syn_m = means_syn
            quats_m = F.normalize(self.quats, dim=-1)
            scales_m = scales
            opacity_m = eff_opacity
            if means_syn_m.ndim == 3 and means_syn_m.shape[0] == 1:
                means_syn_m = means_syn_m[0]
            if quats_m.ndim == 3 and quats_m.shape[0] == 1:
                quats_m = quats_m[0]
            if scales_m.ndim == 3 and scales_m.shape[0] == 1:
                scales_m = scales_m[0]
            if opacity_m.ndim > 1:
                opacity_m = opacity_m.reshape(-1)
            render_colors, alpha, _render_normals, _surf_normals, _render_distort, render_median, _meta = rasterization_2dgs(
                means=means_syn_m,
                quats=quats_m,
                scales=scales_m,
                opacities=opacity_m,
                colors=colors,
                viewmats=view.unsqueeze(0),
                Ks=K_global.unsqueeze(0),
                width=width,
                height=height,
                near_plane=float(near_plane),
                far_plane=float(far_plane),
                sh_degree=None,
                packed=False,
                render_mode="D",
                depth_mode="median",
                median_threshold=self._lidar_median_threshold,
            )
            depth = self._reshape_output(render_median, height, width, 1)
            alpha = self._reshape_output(alpha, height, width, 1)
            return depth, alpha

        tile_size = 16
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii,
            depths,
            tile_size,
            tile_width,
            tile_height,
            packed=False,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics,
            colors,
            eff_opacity,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=None,
            packed=False,
        )
        depth = render_colors[..., 0] / render_alphas[..., 0].clamp_min(1.0e-10)
        depth = depth * (render_alphas[..., 0] > min_alpha).to(depth.dtype)
        return depth, render_alphas[..., 0]

    def _render_lidar_spherical_proxy_ut(
        self,
        means_l: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        lidar_to_world: torch.Tensor,
        width: int,
        height: int,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        near_plane: float,
        far_plane: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rasterize_to_pixels is None or isect_tiles is None or isect_offset_encode is None:
            raise ImportError("gsplat rasterization helpers are required for the spherical proxy backend.")
        if means_l.ndim != 2 or means_l.shape[-1] != 3:
            raise ValueError(f"means_l must be [N,3], got {tuple(means_l.shape)}")

        device = means_l.device
        dtype = means_l.dtype
        eps = 1.0e-6
        min_alpha = 1.0 / 255.0
        eps2d = 0.3

        x = means_l[:, 0]
        y = means_l[:, 1]
        z = means_l[:, 2]
        xy2 = (x.square() + y.square()).clamp_min(eps)
        xy = torch.sqrt(xy2)
        r2 = (xy2 + z.square()).clamp_min(eps)
        r = torch.sqrt(r2)
        az = torch.atan2(y, x)
        el = torch.atan2(z, xy)

        min_el = math.radians(float(vertical_fov_min_deg))
        max_el = math.radians(float(vertical_fov_max_deg))
        span = max(max_el - min_el, eps)

        valid = (r >= float(near_plane)) & (r <= float(far_plane)) & (el >= min_el) & (el <= max_el)

        # UT-based 2D footprint: get the full 2x2 covariance in image space,
        # then use its diagonal for tile bounds and the full inverse for conics.
        mean_uv, cov_uv, valid_mask = self._ut_project_sigmas(
            means_l=means_l,
            scales=scales,
            quats=F.normalize(self.quats, dim=-1),
            width=width,
            height=height,
            vertical_fov_min_deg=vertical_fov_min_deg,
            vertical_fov_max_deg=vertical_fov_max_deg,
            vertical_angles_deg=None,
            near_plane=near_plane,
            far_plane=far_plane,
        )
        valid = valid & valid_mask

        cov_uv = 0.5 * (cov_uv + cov_uv.transpose(-1, -2))
        eye2 = torch.eye(2, device=device, dtype=dtype)
        cov_blur = cov_uv + eps2d * eye2 + eps * eye2
        det_orig = (cov_uv[..., 0, 0] * cov_uv[..., 1, 1] - cov_uv[..., 0, 1] * cov_uv[..., 1, 0]).clamp_min(0.0)
        det_blur = (cov_blur[..., 0, 0] * cov_blur[..., 1, 1] - cov_blur[..., 0, 1] * cov_blur[..., 1, 0]).clamp_min(eps)
        compensation = torch.sqrt(torch.clamp(det_orig / det_blur, min=0.0))
        cov2d_inv = torch.linalg.inv(cov_blur)
        conics = torch.stack(
            [
                cov2d_inv[..., 0, 0],
                cov2d_inv[..., 0, 1],
                cov2d_inv[..., 1, 1],
            ],
            dim=-1,
        )

        extend = torch.full_like(r, 3.33)
        if opacity is not None:
            eff_opacity = opacity * compensation
            extend = torch.minimum(
                extend,
                torch.sqrt(
                    torch.clamp(2.0 * torch.log(torch.clamp(eff_opacity / min_alpha, min=1.0)), min=0.0)
                ),
            )

        cov_diag = torch.diagonal(cov_blur, dim1=-2, dim2=-1)
        radius = torch.ceil(
            torch.minimum(
                extend[..., None] * torch.sqrt(cov_diag.clamp_min(eps)),
                (extend * torch.sqrt(cov_blur[..., 0, 0] + cov_blur[..., 1, 1] + eps)).unsqueeze(-1),
            )
        )
        radii = radius.to(torch.int32)

        valid = valid & (torch.max(radii, dim=-1)[0] > 0)
        mean_uv = torch.where(valid[..., None], mean_uv, torch.zeros_like(mean_uv))
        depths = torch.where(valid, r, torch.zeros_like(r))
        conics = torch.where(valid[..., None], conics, torch.zeros_like(conics))
        radii = torch.where(valid[..., None], radii, torch.zeros_like(radii))
        eff_opacity = torch.where(valid, opacity * compensation, torch.zeros_like(opacity)).unsqueeze(0)

        colors = depths[..., None].unsqueeze(0)
        render_depth_agg = self._lidar_depth_aggregation
        if render_depth_agg != "mean":
            raise NotImplementedError(
                "spherical_proxy_ut currently supports only lidar_depth_aggregation='mean'."
            )

        tile_size = 16
        tile_width = math.ceil(width / float(tile_size))
        tile_height = math.ceil(height / float(tile_size))
        means2d = mean_uv.unsqueeze(0)
        _, isect_ids, flatten_ids = isect_tiles(
            means2d,
            radii.unsqueeze(0),
            depths.unsqueeze(0),
            tile_size,
            tile_width,
            tile_height,
            packed=False,
        )
        isect_offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
        render_colors, render_alphas = rasterize_to_pixels(
            means2d,
            conics.unsqueeze(0),
            colors,
            eff_opacity,
            width,
            height,
            tile_size,
            isect_offsets,
            flatten_ids,
            backgrounds=None,
            packed=False,
        )
        depth = render_colors[..., 0] / render_alphas[..., 0].clamp_min(1.0e-10)
        depth = depth * (render_alphas[..., 0] > min_alpha).to(depth.dtype)
        return depth, render_alphas[..., 0]

    def _render_lidar_gsplat_ut(
        self,
        means: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        width: int,
        height: int,
        lidar_to_world: torch.Tensor,
        near_plane: float,
        far_plane: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rasterization is None:
            raise ImportError("gsplat is required for gsplat-backed LiDAR rendering.")
        if self._lidar_render_config is None:
            raise RuntimeError("Lidar render config is not set.")
        device = means.device
        dtype = means.dtype
        lidar_coeffs = self._get_lidar_coeffs(width=width, height=height, device=device)

        # Native gsplat lidar model uses its own row/column convention and ignores
        # the standard pinhole intrinsics. Use a dummy K with a stable principal point.
        lidar_rows = int(getattr(lidar_coeffs, "n_rows", width))
        lidar_cols = int(getattr(lidar_coeffs, "n_columns", height))
        dummy_k = torch.tensor(
            [
                [float(max(lidar_rows, 1)), 0.0, float(lidar_rows) * 0.5],
                [0.0, float(max(lidar_rows, 1)), float(lidar_cols) * 0.5],
                [0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=dtype,
        ).unsqueeze(0)

        colors = torch.zeros((means.shape[0], 3), device=device, dtype=dtype)
        view = torch.linalg.inv(lidar_to_world).unsqueeze(0)

        render_result = self._call_rasterizer(
            {
                "means": means,
                "quats": F.normalize(self.quats, dim=-1),
                "scales": scales,
                "opacities": opacity,
                "colors": colors,
                "viewmats": view,
                "Ks": dummy_k,
                "width": width,
                "height": height,
                "near_plane": float(near_plane),
                "far_plane": float(far_plane),
                "sh_degree": None,
                "render_mode": "Ed",
                "packed": False,
                "with_ut": True,
                "with_eval3d": True,
                "camera_model": "lidar",
                "lidar_coeffs": lidar_coeffs,
            }
        )
        return self._normalize_native_lidar_render_result(render_result, height, width)

    @staticmethod
    def _normalize_native_lidar_render_result(
        render_result: Any,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(render_result, dict):
            depth = render_result.get("depth")
            if depth is None:
                depth = render_result.get("render")
            if depth is None:
                depth = render_result.get("rgb")
            alpha = render_result.get("alpha")
            if alpha is None:
                alpha = render_result.get("acc")
        elif isinstance(render_result, tuple):
            depth = render_result[0]
            alpha = render_result[1] if len(render_result) > 1 else None
        else:
            raise TypeError(f"Unsupported gsplat render output type: {type(render_result)!r}")

        if depth is None:
            raise ValueError("Native lidar render output does not contain depth data.")
        depth = GaussianSceneModel._reshape_output(depth, height, width, 1)
        if alpha is None:
            alpha = torch.ones_like(depth)
        else:
            alpha = GaussianSceneModel._reshape_output(alpha, height, width, 1)
        return depth, alpha


    def _render_lidar_gsplat_spherical(
        self,
        means_l: torch.Tensor,
        scales: torch.Tensor,
        opacity: torch.Tensor,
        width: int,
        height: int,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        near_plane: float,
        far_plane: float,
        vertical_angles_deg: list[float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rasterization is None:
            raise ImportError("gsplat is required for gsplat-backed LiDAR rendering.")
        if means_l.ndim != 2 or means_l.shape[-1] != 3:
            raise ValueError(f"means_l must be [N,3], got {tuple(means_l.shape)}")

        device = means_l.device
        dtype = means_l.dtype
        N = int(means_l.shape[0])
        x = means_l[:, 0]
        y = means_l[:, 1]
        z = means_l[:, 2]
        xy = torch.sqrt(x.square() + y.square()).clamp_min(1.0e-6)
        r = torch.sqrt(x.square() + y.square() + z.square()).clamp_min(1.0e-6)
        az = torch.atan2(y, x)
        el = torch.atan2(z, xy)
        min_el = math.radians(float(vertical_fov_min_deg))
        max_el = math.radians(float(vertical_fov_max_deg))
        el_span = max(max_el - min_el, 1.0e-6)
        valid = (r >= float(near_plane)) & (r <= float(far_plane)) & (el >= min_el) & (el <= max_el)

        means_p = means_l.clone()
        means_p[~valid] = means_p[~valid] + torch.tensor([0.0, 0.0, 1.0e6], device=device, dtype=dtype)
        u = (az + math.pi) / (2.0 * math.pi) * float(width)
        v = elevation_to_row(
            elevation=el,
            height=height,
            vertical_fov_min_deg=vertical_fov_min_deg,
            vertical_fov_max_deg=vertical_fov_max_deg,
            vertical_angles_deg=vertical_angles_deg,
        )

        denom_xy2 = (x.square() + y.square()).clamp_min(1.0e-6)
        fx = (float(width) / (2.0 * math.pi)) * (r / denom_xy2.sqrt())
        fy = (float(height) / el_span) * (r / (r + 1.0e-6))
        fx = fx.clamp(min=1.0, max=float(width) * 4.0)
        fy = fy.clamp(min=1.0, max=float(height) * 4.0)
        fx_g = float(torch.median(fx).item())
        fy_g = float(torch.median(fy).item())
        K_global = torch.tensor(
            [[fx_g, 0.0, float(width) * 0.5], [0.0, fy_g, float(height) * 0.5], [0.0, 0.0, 1.0]],
            device=device,
            dtype=dtype,
        )
        Xn = (u - K_global[0, 2]) / K_global[0, 0]
        Yn = (v - K_global[1, 2]) / K_global[1, 1]
        means_syn = torch.stack([Xn * r, Yn * r, r], dim=-1)
        colors = torch.zeros((N, 3), device=device, dtype=dtype)
        view = torch.eye(4, device=device, dtype=dtype)
        depth_agg = self._lidar_depth_aggregation
        if depth_agg == "median":
            if rasterization_2dgs is None:
                raise ImportError("gsplat 2DGS rasterization is required for LiDAR median depth.")
            _render, alpha, _render_normals, _surf_normals, _render_distort, render_median, _meta = rasterization_2dgs(
                means=means_syn,
                quats=F.normalize(self.quats, dim=-1),
                scales=scales,
                opacities=opacity,
                colors=colors,
                viewmats=view.unsqueeze(0),
                Ks=K_global.unsqueeze(0),
                width=width,
                height=height,
                near_plane=float(near_plane),
                far_plane=float(far_plane),
                sh_degree=None,
                packed=False,
                render_mode="D",
                depth_mode="median",
                median_threshold=self._lidar_median_threshold,
            )
            depth = self._reshape_output(render_median, height, width, 1)
            alpha = self._reshape_output(alpha, height, width, 1)
        else:
            render, alpha, _meta = self._call_rasterizer(
                {
                    "means": means_syn,
                    "quats": F.normalize(self.quats, dim=-1),
                    "scales": scales,
                    "opacities": opacity,
                    "colors": colors,
                    "viewmats": view.unsqueeze(0),
                    "Ks": K_global.unsqueeze(0),
                    "width": width,
                    "height": height,
                    "near_plane": float(near_plane),
                    "far_plane": float(far_plane),
                    "sh_degree": None,
                    "render_mode": "ED",
                    "packed": False,
                    "with_eval3d": True,
                }
            )
            while render.ndim > 4 and render.shape[0] == 1:
                render = render[0]
            while alpha.ndim > 4 and alpha.shape[0] == 1:
                alpha = alpha[0]
            if render.ndim == 4:
                render = render[0]
            if render.ndim == 4 and render.shape[-1] >= 4:
                depth = render[..., 3].permute(2, 0, 1)
            elif render.ndim == 3 and render.shape[-1] >= 4:
                depth = render[..., 3].unsqueeze(0)
            else:
                depth = render
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            if depth.shape[-2:] != (height, width):
                depth = render[..., 3].unsqueeze(0) if render.ndim >= 3 and render.shape[-1] >= 4 else depth
            if alpha.ndim == 4:
                alpha = alpha[0]
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth.squeeze(-1)
            if alpha.ndim == 3 and alpha.shape[-1] == 1:
                alpha = alpha.squeeze(-1)
            if depth.ndim == 3 and depth.shape[-2:] == (width, height):
                depth = depth.permute(0, 2, 1)
            if alpha.ndim == 3 and alpha.shape[-2:] == (width, height):
                alpha = alpha.permute(0, 2, 1)
            if depth.ndim == 3 and alpha.ndim == 3 and depth.shape[-2:] != alpha.shape[-2:]:
                alpha = alpha.transpose(-1, -2)
            if alpha.ndim == 2:
                alpha = alpha.unsqueeze(0)
            depth = depth * (alpha > (1.0 / 255.0)).to(depth.dtype)
        return depth, alpha

    def _ut_project_sigmas(
        self,
        means_l: torch.Tensor,
        scales: torch.Tensor,
        quats: torch.Tensor,
        width: int,
        height: int,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        vertical_angles_deg: list[float] | None,
        near_plane: float,
        far_plane: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Unscented transform of 3D Gaussian into image space mean/covariance."""
        device = means_l.device
        dtype = means_l.dtype
        N = means_l.shape[0]
        D = 3
        alpha = self._lidar_ut_alpha
        beta = self._lidar_ut_beta
        kappa = self._lidar_ut_kappa
        lam = alpha * alpha * (D + kappa) - D
        if self._lidar_ut_delta > 0:
            delta = self._lidar_ut_delta
        else:
            delta = math.sqrt(max(D + lam, 1.0e-6))
        w0_m = lam / (D + lam)
        w0_c = w0_m + (1.0 - alpha * alpha + beta)
        wi = 1.0 / (2.0 * (D + lam))

        # rotation matrices from quats
        q = F.normalize(quats, dim=-1)
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        # column vectors of rotation matrix
        r0 = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)], dim=-1)
        r1 = torch.stack([2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)], dim=-1)
        r2 = torch.stack([2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)], dim=-1)

        # sigma points: mean +/− delta * scale_i * rot_i
        sigmas = [means_l]
        for i, ri in enumerate([r0, r1, r2]):
            d = (delta * scales[:, i]).unsqueeze(-1) * ri
            sigmas.append(means_l + d)
            sigmas.append(means_l - d)
        sigma_pts = torch.stack(sigmas, dim=1)  # [N, 2D+1, 3]

        # project to (u,v)
        M = sigma_pts.reshape(-1, 3)
        x, y, z = M[:, 0], M[:, 1], M[:, 2]
        xy = torch.sqrt(x * x + y * y).clamp_min(1.0e-6)
        r = torch.sqrt(x * x + y * y + z * z).clamp_min(1.0e-6)
        az = torch.atan2(y, x)
        el = torch.atan2(z, xy)
        min_el = math.radians(float(vertical_fov_min_deg))
        max_el = math.radians(float(vertical_fov_max_deg))
        valid = (r >= near_plane) & (r <= far_plane) & (el >= min_el) & (el <= max_el)

        # Unwrap azimuth around mean to handle -pi/pi discontinuity for UT sigma points
        az0 = torch.atan2(means_l[:, 1], means_l[:, 0])  # [N]
        az0 = az0.repeat_interleave(2 * D + 1)
        d = az - az0
        d = (d + math.pi) % (2.0 * math.pi) - math.pi
        az = az0 + d

        u = (az + math.pi) / (2.0 * math.pi) * float(width)
        v = elevation_to_row(
            elevation=el,
            height=height,
            vertical_fov_min_deg=vertical_fov_min_deg,
            vertical_fov_max_deg=vertical_fov_max_deg,
            vertical_angles_deg=vertical_angles_deg,
        )

        uv = torch.stack([u, v], dim=-1).reshape(N, 2 * D + 1, 2)
        valid = valid.reshape(N, 2 * D + 1)

        # Image margin validity (3DGUT-style)
        margin = self._lidar_ut_in_image_margin_factor
        if margin > 0:
            umin, umax = -margin * float(width), (1.0 + margin) * float(width)
            vmin, vmax = -margin * float(height), (1.0 + margin) * float(height)
            in_img = (uv[..., 0] >= umin) & (uv[..., 0] <= umax) & (uv[..., 1] >= vmin) & (uv[..., 1] <= vmax)
            valid = valid & in_img

        # validity policy
        if self._lidar_ut_require_all_sigma_points_valid:
            valid_mask = valid.all(dim=1)
        else:
            valid_mask = valid.any(dim=1)

        # compute mean
        w0 = w0_m
        mean_uv = w0 * uv[:, 0]
        mean_uv += wi * uv[:, 1:].sum(dim=1)

        # covariance
        d0 = uv[:, 0] - mean_uv
        cov = w0_c * torch.einsum("ni,nj->nij", d0, d0)
        d = uv[:, 1:] - mean_uv.unsqueeze(1)
        cov += wi * torch.einsum("nki,nkj->nij", d, d)
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        return mean_uv, cov, valid_mask

    def parameter_groups(self, lr_config: Any) -> list[dict[str, Any]]:
        groups = [
            {"params": [self.means], "lr": lr_config.lr_mean, "name": "means"},
            {"params": [self.sh_coeffs], "lr": lr_config.lr_color, "name": "sh_coeffs"},
            {"params": [self.log_scales], "lr": lr_config.lr_scale, "name": "log_scales"},
            {"params": [self.quats], "lr": lr_config.lr_rotation, "name": "quats"},
        ]
        if self.use_separate_opacity:
            groups.extend(
                [
                    {
                        "params": [self.opacity_camera],
                        "lr": lr_config.lr_opacity,
                        "name": "opacity_camera",
                    },
                    {
                        "params": [self.opacity_lidar],
                        "lr": lr_config.lr_opacity,
                        "name": "opacity_lidar",
                    },
                ]
            )
        else:
            groups.append({"params": [self.opacities], "lr": lr_config.lr_opacity, "name": "opacities"})

        return groups

    def get_opacity_regularization_loss(self) -> torch.Tensor:
        """
        Compute opacity consistency regularization loss.

        L_opacity = Σ |σc - σL|

        This ensures the two opacity parameters remain consistent while allowing
        them to differ where necessary for accurate modeling.
        """
        if not self.use_separate_opacity:
            return torch.tensor(0.0, device=self.means.device)

        opacity_camera = self.opacity_camera.clamp(0.0, 1.0)
        opacity_lidar = self.opacity_lidar.clamp(0.0, 1.0)
        return (opacity_camera - opacity_lidar).abs().mean()


    @staticmethod
    def _transform_points(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
        homogeneous = torch.cat([points, ones], dim=-1)
        return (transform @ homogeneous.T).T[:, :3]

    @staticmethod
    def _random_quaternions(count: int) -> torch.Tensor:
        values = torch.randn(count, 4)
        return F.normalize(values, dim=-1)

    @staticmethod
    def _inverse_sigmoid(value: float) -> float:
        value = min(max(value, 1.0e-4), 1.0 - 1.0e-4)
        return torch.logit(torch.tensor(value)).item()

    @staticmethod
    @staticmethod
    def _normalize_render_result(
        render_result: Any,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        extras: dict[str, Any] = {}

        if isinstance(render_result, dict):
            rgb = render_result.get("render") or render_result.get("rgb")
            depth = render_result.get("depth")
            alpha = render_result.get("alpha") or render_result.get("acc")
            extras = {k: v for k, v in render_result.items() if k not in {"render", "rgb", "depth", "alpha", "acc"}}
        elif isinstance(render_result, tuple):
            rgb = render_result[0]
            depth = render_result[1] if len(render_result) > 1 else None
            alpha = None
            if len(render_result) > 2 and isinstance(render_result[2], dict):
                extras = render_result[2]
            elif len(render_result) > 2:
                alpha = render_result[2]
                if len(render_result) > 3 and isinstance(render_result[3], dict):
                    extras = render_result[3]
        else:
            raise TypeError(f"Unsupported gsplat render output type: {type(render_result)!r}")

        if rgb is None:
            raise ValueError("gsplat render output does not contain RGB data.")

        rgb = GaussianSceneModel._reshape_output(rgb, height, width, 3)
        depth = GaussianSceneModel._reshape_output(depth, height, width, 1) if depth is not None else torch.zeros(
            1, height, width, device=rgb.device, dtype=rgb.dtype
        )
        alpha = GaussianSceneModel._reshape_output(alpha, height, width, 1) if alpha is not None else torch.ones(
            1, height, width, device=rgb.device, dtype=rgb.dtype
        )
        return rgb, depth, alpha, extras

    @staticmethod
    def _reshape_output(value: torch.Tensor, height: int, width: int, channels: int) -> torch.Tensor:
        if value.ndim == 4:
            value = value[0]
        if value.ndim == 3:
            if value.shape[0] == channels and value.shape[1] == height and value.shape[2] == width:
                pass
            elif value.shape[0] == height and value.shape[1] == width and value.shape[2] == channels:
                value = value.permute(2, 0, 1)
            elif value.shape[0] == channels and value.shape[1] == width and value.shape[2] == height:
                value = value.permute(0, 2, 1)
            elif value.shape[0] == width and value.shape[1] == height and value.shape[2] == channels:
                value = value.permute(2, 1, 0)
            elif value.shape[-2:] == (height, width):
                if value.shape[0] != channels:
                    raise ValueError(f"Unexpected rendered tensor shape: {tuple(value.shape)}")
            elif value.shape[-2:] == (width, height):
                if value.shape[0] == channels:
                    value = value.transpose(-1, -2)
                else:
                    raise ValueError(f"Unexpected rendered tensor shape: {tuple(value.shape)}")
            else:
                raise ValueError(f"Unexpected rendered tensor shape: {tuple(value.shape)}")
        elif value.ndim == 2 and channels == 1 and value.shape == (height, width):
            value = value.unsqueeze(0)
        elif value.ndim == 2 and channels == 1 and value.shape == (width, height):
            value = value.t().unsqueeze(0)
        else:
            raise ValueError(f"Unexpected rendered tensor shape: {tuple(value.shape)}")
        return value
