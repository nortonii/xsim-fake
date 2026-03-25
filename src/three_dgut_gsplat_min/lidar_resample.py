from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def resample_depth_to_shape(
    depth: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Resize a depth map to a target shape using nearest valid-neighbor semantics.

    Invalid pixels remain zero; valid pixels are interpolated with bilinear resize
    after replacing zeros with NaN-safe sentinels.
    """
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth = depth.unsqueeze(1)
    elif depth.ndim != 4:
        raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

    x = depth.clone()
    valid = x > 0
    x[~valid] = float("nan")
    x = torch.nan_to_num(x, nan=0.0)
    x = torch.nn.functional.interpolate(x, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return x[:, 0]


def resample_depth_np(depth: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    import cv2  # type: ignore

    if depth.ndim != 2:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")
    resized = cv2.resize(depth.astype(np.float32), (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32, copy=False)


def warp_depth_to_vertical_angles(
    depth: torch.Tensor,
    target_vertical_angles_deg: list[float] | None,
    source_vertical_fov_min_deg: float,
    source_vertical_fov_max_deg: float,
) -> torch.Tensor:
    """Warp a depth image rendered on a uniform vertical grid to a target angle table.

    The source image is assumed to have rows uniformly spaced across the source FOV.
    The output keeps the same width and uses one output row per target angle.
    """
    if target_vertical_angles_deg is None or len(target_vertical_angles_deg) == 0:
        return depth

    if depth.ndim == 2:
        depth_4d = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        depth_4d = depth.unsqueeze(1)
    elif depth.ndim == 4:
        depth_4d = depth
    else:
        raise ValueError(f"Unexpected depth shape: {tuple(depth.shape)}")

    b, c, src_h, src_w = depth_4d.shape
    if c != 1:
        raise ValueError(f"Expected a single-channel depth map, got shape {tuple(depth.shape)}")

    device = depth_4d.device
    dtype = depth_4d.dtype
    target_angles = torch.as_tensor(target_vertical_angles_deg, device=device, dtype=dtype).flatten()
    target_angles = torch.sort(target_angles, descending=True)[0]
    target_h = int(target_angles.numel())
    if target_h == 0:
        return depth

    min_el = torch.tensor(np.deg2rad(source_vertical_fov_min_deg), device=device, dtype=dtype)
    max_el = torch.tensor(np.deg2rad(source_vertical_fov_max_deg), device=device, dtype=dtype)
    span = (max_el - min_el).clamp_min(torch.tensor(1.0e-6, device=device, dtype=dtype))
    target_el = torch.deg2rad(target_angles)
    src_row = (max_el - target_el) / span * float(max(src_h - 1, 1))
    src_row = src_row.clamp(0.0, float(max(src_h - 1, 0)))

    if src_h <= 1:
        return depth

    y_norm = src_row / float(src_h - 1) * 2.0 - 1.0
    x_norm = torch.linspace(-1.0, 1.0, src_w, device=device, dtype=dtype)
    grid_y = y_norm[:, None].expand(target_h, src_w)
    grid_x = x_norm[None, :].expand(target_h, src_w)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)

    warped = F.grid_sample(
        depth_4d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    if depth.ndim == 2:
        return warped[0, 0]
    if depth.ndim == 3:
        return warped[:, 0]
    return warped
