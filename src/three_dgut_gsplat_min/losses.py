from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LossBreakdown:
    total: torch.Tensor
    terms: dict[str, float]
    raw_terms: dict[str, torch.Tensor] | None = None


def _gaussian_window(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum().clamp_min(1.0e-12)
    return g


def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM for [3,H,W] images in [0,1]. Returns scalar tensor."""
    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(f"SSIM expects [C,H,W], got {tuple(img1.shape)} and {tuple(img2.shape)}")
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(img1.shape)} vs {tuple(img2.shape)}")

    # Parameters used by common implementations.
    window_size = 11
    sigma = 1.5
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    device = img1.device
    dtype = img1.dtype

    # Build separable Gaussian kernel.
    g = _gaussian_window(window_size, sigma, device=device, dtype=dtype)
    window_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]

    # Per-channel conv.
    C = img1.shape[0]
    w = window_2d.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    x1 = img1.unsqueeze(0)
    x2 = img2.unsqueeze(0)

    mu1 = F.conv2d(x1, w, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(x2, w, padding=window_size // 2, groups=C)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x1 * x1, w, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, w, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(x1 * x2, w, padding=window_size // 2, groups=C) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)).clamp_min(1.0e-12)
    return ssim_map.mean()


def _ssim_masked(img1: torch.Tensor, img2: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute SSIM for [3,H,W] images in [0,1] with a [H,W] mask (1=valid). Returns scalar tensor."""
    if img1.ndim != 3 or img2.ndim != 3:
        raise ValueError(f"SSIM expects [C,H,W], got {tuple(img1.shape)} and {tuple(img2.shape)}")
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM shape mismatch: {tuple(img1.shape)} vs {tuple(img2.shape)}")
    if mask.ndim != 2:
        raise ValueError(f"Mask expects [H,W], got {tuple(mask.shape)}")

    window_size = 11
    sigma = 1.5
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    device = img1.device
    dtype = img1.dtype

    g = _gaussian_window(window_size, sigma, device=device, dtype=dtype)
    window_2d = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]

    C = img1.shape[0]
    w = window_2d.repeat(C, 1, 1, 1)  # [C,1,ws,ws]

    # Apply mask: set invalid regions to 0
    mask_3ch = mask.unsqueeze(0).expand_as(img1)  # [3, H, W]
    img1_masked = img1 * mask_3ch
    img2_masked = img2 * mask_3ch

    x1 = img1_masked.unsqueeze(0)
    x2 = img2_masked.unsqueeze(0)
    m = mask_3ch.unsqueeze(0)  # [1, 3, H, W]

    mu1 = F.conv2d(x1, w, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(x2, w, padding=window_size // 2, groups=C)
    mu_mask = F.conv2d(m.float(), w, padding=window_size // 2, groups=C)

    # Normalize by mask weight
    mu_mask = mu_mask.clamp_min(1.0e-6)
    mu1 = mu1 / mu_mask
    mu2 = mu2 / mu_mask

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x1 * x1, w, padding=window_size // 2, groups=C) / mu_mask - mu1_sq
    sigma2_sq = F.conv2d(x2 * x2, w, padding=window_size // 2, groups=C) / mu_mask - mu2_sq
    sigma12 = F.conv2d(x1 * x2, w, padding=window_size // 2, groups=C) / mu_mask - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)).clamp_min(1.0e-12)

    # Average only over valid regions
    valid_ssim = ssim_map * mask_3ch.unsqueeze(0)
    return valid_ssim.sum() / mask_3ch.sum().clamp_min(1.0)


class JointLoss:
    """Joint loss for RGB camera and LiDAR training."""

    def __init__(self, config: Any, dataset_config: Any | None = None) -> None:
        self.config = config
        self.dataset_config = dataset_config

    @staticmethod
    def _build_lidar_row_weights(
        height: int,
        vertical_angles_deg: list[float] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if vertical_angles_deg is None or len(vertical_angles_deg) < 2:
            return None
        angles = torch.as_tensor(vertical_angles_deg, device=device, dtype=dtype).flatten()
        if int(angles.numel()) != int(height):
            return None
        angles = torch.sort(angles, descending=False)[0]

        # Estimate per-ring solid-angle proxy by integrating over angular bins.
        # We only change sampling density, not the elevation range.
        edges = torch.empty((angles.numel() + 1,), device=device, dtype=dtype)
        edges[1:-1] = 0.5 * (angles[:-1] + angles[1:])
        first_gap = (angles[1] - angles[0]).clamp_min(1.0e-6)
        last_gap = (angles[-1] - angles[-2]).clamp_min(1.0e-6)
        edges[0] = angles[0] - 0.5 * first_gap
        edges[-1] = angles[-1] + 0.5 * last_gap

        row_width = (edges[1:] - edges[:-1]).abs().clamp_min(1.0e-6)
        # Solid-angle proxy for a horizontal ring: dΩ ≈ cos(el) d_el d_az.
        row_center = angles
        row_weight = row_width * torch.cos(row_center).abs().clamp_min(1.0e-6)
        row_weight = row_weight / row_weight.mean().clamp_min(1.0e-6)
        return row_weight

    def _lidar_pixel_weights(self, target_lidar: torch.Tensor) -> torch.Tensor | None:
        sampling = str(getattr(self.config, "lidar_loss_sampling", "uniform")).strip().lower()
        if sampling != "angle_table":
            return None
        if self.dataset_config is None:
            return None
        angles = getattr(self.dataset_config, "lidar_vertical_angles_deg", None)
        if angles is None or len(angles) == 0:
            return None

        height = int(target_lidar.shape[-2])
        row_weights = self._build_lidar_row_weights(
            height=height,
            vertical_angles_deg=list(angles),
            device=target_lidar.device,
            dtype=target_lidar.dtype,
        )
        if row_weights is None:
            return None

        weights = row_weights.view(1, height, 1).expand_as(target_lidar)
        return weights

    def __call__(self, batch: dict[str, Any], render_output: Any, model: Any) -> LossBreakdown:
        losses: dict[str, torch.Tensor] = {}

        # RGB reconstruction: L = (1-λ)*L1 + λ*(1-SSIM)
        pred_rgb = render_output.rgb.rgb
        gt_rgb = batch["rgb"]
        lmbd = float(getattr(self.config, "rgb_ssim_lambda", 0.0) or 0.0)
        lmbd = max(0.0, min(1.0, lmbd))

        # Apply dynamic mask: exclude dynamic regions from RGB loss
        dynamic_mask = batch.get("dynamic_mask", None)
        if dynamic_mask is not None:
            # dynamic_mask: [1, H, W], 1=dynamic, 0=static
            # Create static mask: 1=static, 0=dynamic
            static_mask = (dynamic_mask < 0.5).float()
            # Expand to [3, H, W] for RGB
            static_mask = static_mask.expand_as(gt_rgb)
            # Apply mask to L1 loss
            if static_mask.sum() > 0:
                l1 = (pred_rgb - gt_rgb).abs() * static_mask
                l1 = l1.sum() / static_mask.sum().clamp_min(1.0)
            else:
                l1 = pred_rgb.new_tensor(0.0)
        else:
            l1 = F.l1_loss(pred_rgb, gt_rgb)

        if lmbd > 0.0:
            if dynamic_mask is not None:
                static_mask_2d = (dynamic_mask < 0.5).squeeze(0)  # [H, W]
                if static_mask_2d.sum() > 0:
                    ssim_val = _ssim_masked(pred_rgb, gt_rgb, static_mask_2d)
                else:
                    ssim_val = pred_rgb.new_tensor(1.0)
            else:
                ssim_val = _ssim(pred_rgb, gt_rgb)
            ssim_loss = (1.0 - ssim_val).clamp(min=0.0, max=1.0)
        else:
            ssim_loss = pred_rgb.new_tensor(0.0)

        losses["rgb_l1"] = l1 * (1.0 - lmbd) * self.config.rgb_l1_weight
        losses["rgb_ssim"] = ssim_loss * lmbd * self.config.rgb_l1_weight

        target_lidar = batch["lidar_depth"]
        pred_lidar = render_output.lidar.depth
        valid_mask = target_lidar > 0
        lidar_weights = self._lidar_pixel_weights(target_lidar)

        # LiDAR depth loss
        if valid_mask.any():
            pred_v = pred_lidar[valid_mask]
            tgt_v = target_lidar[valid_mask]
            weight_v = lidar_weights[valid_mask] if lidar_weights is not None else None
            loss_type = str(getattr(self.config, "lidar_loss_type", "smooth_l1")).lower()
            if loss_type in ("pearson", "corr", "correlation"):
                # Pearson correlation loss: 1 - corr(pred, gt)
                finite = torch.isfinite(pred_v) & torch.isfinite(tgt_v)
                if not bool(finite.any()):
                    corr = pred_v.new_tensor(0.0)
                else:
                    pred_v = pred_v[finite]
                    tgt_v = tgt_v[finite]
                    if weight_v is not None:
                        weight_v = weight_v[finite]

                    if weight_v is None:
                        pred_c = pred_v - pred_v.mean()
                        tgt_c = tgt_v - tgt_v.mean()
                        pred_norm = pred_c.norm()
                        tgt_norm = tgt_c.norm()
                        if (not torch.isfinite(pred_norm)) or (not torch.isfinite(tgt_norm)) or pred_norm <= 1.0e-6 or tgt_norm <= 1.0e-6:
                            corr = pred_v.new_tensor(0.0)
                        else:
                            corr = (pred_c * tgt_c).sum() / (pred_norm * tgt_norm)
                    else:
                        w = weight_v.clamp_min(1.0e-6)
                        w_sum = w.sum().clamp_min(1.0e-6)
                        pred_mean = (w * pred_v).sum() / w_sum
                        tgt_mean = (w * tgt_v).sum() / w_sum
                        pred_c = pred_v - pred_mean
                        tgt_c = tgt_v - tgt_mean
                        cov = (w * pred_c * tgt_c).sum()
                        var_p = (w * pred_c.square()).sum()
                        var_t = (w * tgt_c.square()).sum()
                        if (
                            (not torch.isfinite(cov))
                            or (not torch.isfinite(var_p))
                            or (not torch.isfinite(var_t))
                            or var_p <= 1.0e-6
                            or var_t <= 1.0e-6
                        ):
                            corr = pred_v.new_tensor(0.0)
                        else:
                            corr = cov / torch.sqrt(var_p * var_t)
                if not torch.isfinite(corr):
                    corr = pred_v.new_tensor(0.0)
                losses["lidar_depth"] = (1.0 - corr) * self.config.lidar_depth_weight
            else:
                if weight_v is None:
                    losses["lidar_depth"] = F.smooth_l1_loss(pred_v, tgt_v) * self.config.lidar_depth_weight
                else:
                    per_pixel = F.smooth_l1_loss(pred_v, tgt_v, reduction="none")
                    losses["lidar_depth"] = (per_pixel * weight_v).sum() / weight_v.sum().clamp_min(1.0e-6)
                    losses["lidar_depth"] = losses["lidar_depth"] * self.config.lidar_depth_weight
        else:
            losses["lidar_depth"] = pred_lidar.new_tensor(0.0)

        # Opacity regularization (XSIM contribution)
        if hasattr(model, "get_opacity_regularization_loss"):
            opacity_reg = model.get_opacity_regularization_loss()
        else:
            if hasattr(model, "opacities"):
                opacity_reg = model.opacities.mean()
            else:
                opacity_reg = torch.sigmoid(model.opacity_logits).mean()
        losses["opacity_reg"] = opacity_reg * self.config.opacity_reg_weight

        # Encourage LiDAR opacity to be near 0 or 1 (binarize)
        bin_w = float(getattr(self.config, "lidar_opacity_binarize_weight", 0.0) or 0.0)
        if bin_w > 0.0 and hasattr(model, "get_lidar_opacity"):
            p = model.get_lidar_opacity().clamp(0.0, 1.0)
            # 4*p*(1-p) peaks at 0.5; minimizing pushes to 0 or 1.
            losses["lidar_opacity_binarize"] = (4.0 * p * (1.0 - p)).mean() * bin_w
        else:
            losses["lidar_opacity_binarize"] = pred_lidar.new_tensor(0.0)

        # Scale regularization
        # Conventional 3DGS uses log-scales; penalize the positive scales.
        losses["scale_reg"] = torch.exp(model.log_scales).mean() * self.config.scale_reg_weight

        total = sum(losses.values(), torch.tensor(0.0, device=next(iter(losses.values())).device))
        return LossBreakdown(
            total=total,
            terms={name: float(value.detach().item()) for name, value in losses.items()},
            raw_terms=losses,
        )
