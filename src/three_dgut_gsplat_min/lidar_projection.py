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


def elevation_to_row_value(
    elevation: np.ndarray | torch.Tensor,
    vertical_angles_deg: list[float] | None,
    row_values: list[float] | None,
) -> np.ndarray | torch.Tensor:
    if (
        vertical_angles_deg is None
        or row_values is None
        or len(vertical_angles_deg) == 0
        or len(vertical_angles_deg) != len(row_values)
    ):
        if isinstance(elevation, np.ndarray):
            return np.zeros_like(elevation, dtype=np.float32)
        return torch.zeros_like(elevation)

    if isinstance(elevation, np.ndarray):
        angles = np.asarray(vertical_angles_deg, dtype=np.float32)
        values = np.asarray(row_values, dtype=np.float32)
        order = np.argsort(angles)[::-1]
        angles = np.radians(angles[order])
        values = values[order]
        idx = np.searchsorted(-angles, -elevation)
        idx = np.clip(idx, 1, len(angles) - 1)
        a0 = angles[idx - 1]
        a1 = angles[idx]
        v0 = values[idx - 1]
        v1 = values[idx]
        t = (a0 - elevation) / np.clip(a0 - a1, 1.0e-6, None)
        return (1.0 - t) * v0 + t * v1

    angles = torch.as_tensor(vertical_angles_deg, device=elevation.device, dtype=elevation.dtype)
    values = torch.as_tensor(row_values, device=elevation.device, dtype=elevation.dtype)
    order = torch.argsort(angles, descending=True)
    angles = torch.deg2rad(angles[order])
    values = values[order]
    idx = torch.searchsorted(-angles, -elevation)
    idx = torch.clamp(idx, min=1, max=angles.numel() - 1)
    a0 = angles[idx - 1]
    a1 = angles[idx]
    v0 = values[idx - 1]
    v1 = values[idx]
    t = (a0 - elevation) / (a0 - a1).clamp_min(1.0e-6)
    return (1.0 - t) * v0 + t * v1


def assign_ring_and_beam_ids(
    points_lidar: np.ndarray,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    vertical_fov_min_deg: float,
    vertical_fov_max_deg: float,
    vertical_angles_deg: list[float] | None = None,
    row_azimuth_offsets_deg: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if points_lidar.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    x = points_lidar[:, 0]
    y = points_lidar[:, 1]
    z = points_lidar[:, 2]
    xy = np.sqrt(x * x + y * y)
    xy = np.clip(xy, 1.0e-6, None)
    ranges = np.sqrt(x * x + y * y + z * z)
    ranges = np.clip(ranges, 1.0e-6, None)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, xy)

    min_el = math.radians(vertical_fov_min_deg)
    max_el = math.radians(vertical_fov_max_deg)
    valid = (ranges >= near_plane) & (ranges <= far_plane) & (elevation >= min_el) & (elevation <= max_el)
    if not np.any(valid):
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )

    ranges = ranges[valid]
    azimuth = azimuth[valid]
    elevation = elevation[valid]

    if vertical_angles_deg is None or len(vertical_angles_deg) != int(height):
        ring_id = np.floor(
            elevation_to_row(
                elevation=elevation,
                height=height,
                vertical_fov_min_deg=vertical_fov_min_deg,
                vertical_fov_max_deg=vertical_fov_max_deg,
                vertical_angles_deg=vertical_angles_deg,
            )
        ).astype(np.int64)
        ring_id = np.clip(ring_id, 0, int(height) - 1)
        residual_deg = np.zeros_like(ranges, dtype=np.float32)
    else:
        row_angles_rad = np.radians(np.asarray(vertical_angles_deg, dtype=np.float32))
        diff = np.abs(elevation[:, None] - row_angles_rad[None, :])
        ring_id = np.argmin(diff, axis=1).astype(np.int64)
        residual_deg = np.degrees(diff[np.arange(diff.shape[0]), ring_id]).astype(np.float32)

    if row_azimuth_offsets_deg is not None and len(row_azimuth_offsets_deg) == int(height):
        offsets_rad = np.radians(np.asarray(row_azimuth_offsets_deg, dtype=np.float32))
        azimuth = azimuth - offsets_rad[ring_id]
    azimuth = (azimuth + math.pi) % (2.0 * math.pi) - math.pi
    beam_id = np.floor((azimuth + math.pi) / (2.0 * math.pi) * float(width)).astype(np.int64)
    beam_id = np.mod(beam_id, int(width))
    return ring_id, beam_id, ranges.astype(np.float32), residual_deg


def points_to_angle_table_depth(
    points_lidar: np.ndarray,
    width: int,
    height: int,
    near_plane: float,
    far_plane: float,
    vertical_fov_min_deg: float,
    vertical_fov_max_deg: float,
    vertical_angles_deg: list[float] | None = None,
    row_azimuth_offsets_deg: list[float] | None = None,
) -> torch.Tensor:
    depth = np.zeros((height, width), dtype=np.float32)
    ring_id, beam_id, ranges, _residual_deg = assign_ring_and_beam_ids(
        points_lidar=points_lidar,
        width=width,
        height=height,
        near_plane=near_plane,
        far_plane=far_plane,
        vertical_fov_min_deg=vertical_fov_min_deg,
        vertical_fov_max_deg=vertical_fov_max_deg,
        vertical_angles_deg=vertical_angles_deg,
        row_azimuth_offsets_deg=row_azimuth_offsets_deg,
    )
    if ranges.size == 0:
        return torch.from_numpy(depth).unsqueeze(0)

    flat_idx = ring_id * int(width) + beam_id
    order = np.argsort(ranges)
    flat_idx = flat_idx[order]
    ranges_sorted = ranges[order]
    seen = np.zeros(int(height) * int(width), dtype=bool)
    depth_flat = depth.reshape(-1)
    for idx, r in zip(flat_idx, ranges_sorted, strict=False):
        if not seen[idx]:
            depth_flat[idx] = r
            seen[idx] = True
    return torch.from_numpy(depth).unsqueeze(0)
