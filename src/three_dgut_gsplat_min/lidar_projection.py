from __future__ import annotations

import math

import numpy as np
import torch


def elevation_to_row(
    elevation: np.ndarray | torch.Tensor,
    height: int,
    vertical_fov_min_deg: float,
    vertical_fov_max_deg: float,
    vertical_angles_deg: list[float] | None = None,
) -> np.ndarray | torch.Tensor:
    min_el = math.radians(vertical_fov_min_deg)
    max_el = math.radians(vertical_fov_max_deg)

    if vertical_angles_deg is None or len(vertical_angles_deg) == 0:
        return (max_el - elevation) / max(max_el - min_el, 1.0e-6) * float(height - 1)

    if isinstance(elevation, np.ndarray):
        angles = np.sort(np.radians(np.asarray(vertical_angles_deg, dtype=np.float32)))[::-1]
        idx = np.searchsorted(-angles, -elevation)
        idx = np.clip(idx, 1, len(angles) - 1)
        a0 = angles[idx - 1]
        a1 = angles[idx]
        t = (a0 - elevation) / np.clip(a0 - a1, 1.0e-6, None)
        return idx - 1 + t

    angles = torch.tensor(vertical_angles_deg, device=elevation.device, dtype=elevation.dtype)
    angles = torch.sort(angles, descending=True)[0]
    idx = torch.searchsorted(-angles, -elevation)
    idx = torch.clamp(idx, min=1, max=angles.numel() - 1)
    a0 = angles[idx - 1]
    a1 = angles[idx]
    t = (a0 - elevation) / (a0 - a1).clamp_min(1.0e-6)
    return idx.to(elevation.dtype) - 1 + t
