from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

try:
    import gsplat
    from gsplat.cuda._wrapper import SpinningDirection
except Exception as exc:  # pragma: no cover
    gsplat = None
    SpinningDirection = None  # type: ignore[assignment]
    GSPLAT_IMPORT_ERROR = exc
else:
    GSPLAT_IMPORT_ERROR = None


# Velodyne HDL-64E: 64-line lidar with non-uniform vertical angle distribution
HDL64E_VERT_DEG = [
    -6.98, -6.62, 0.51, 0.81, -6.33, -5.98, -8.31, -8.01, -5.61, -5.26, -7.64, -7.3,
    -2.89, -2.56, -4.94, -4.59, -2.23, -1.88, -4.29, -3.94, -1.55, -1.18, -3.59, -3.24,
    1.14, 1.52, -0.85, -0.55, 1.84, 2.22, -0.15, 0.13, -22.4, -22.1, -11.25, -10.79,
    -21.59, -21.11, -24.61, -24.22, -20.33, -19.85, -23.57, -22.88, -16.32, -15.94,
    -19.16, -18.98, -15.41, -14.89, -18.5, -17.99, -14.29, -13.78, -17.39, -16.79,
    -10.13, -9.7, -13.07, -12.83, -9.27, -8.81, -12.34, -11.86,
]

# Waymo Open Dataset: 64-line lidar (top lidar) with non-uniform vertical angle distribution
# FOV: -17.6° to +2.4°, with denser coverage near the horizon
# Reference: Waymo Open Dataset
WAYMO_TOP_VERT_DEG = [
    2.4000, 2.1000, 1.8000, 1.5000, 1.2000, 0.9000, 0.6000, 0.3000,
    0.0000, -0.3000, -0.6000, -0.9000, -1.2000, -1.5000, -1.8000, -2.1000,
    -2.4000, -2.7000, -3.0000, -3.3000, -3.6000, -3.9000, -4.2000, -4.5000,
    -4.8000, -5.1000, -5.4000, -5.7000, -6.0000, -6.3000, -6.6000, -6.9000,
    -7.2000, -7.5000, -7.8000, -8.1000, -8.4000, -8.7000, -9.0000, -9.3000,
    -9.6000, -9.9000, -10.2000, -10.5000, -10.8000, -11.1000, -11.4000, -11.7000,
    -12.0000, -12.3000, -12.6000, -12.9000, -13.2000, -13.5000, -13.8000, -14.1000,
    -14.4000, -14.7000, -15.0000, -15.3000, -15.6000, -15.9000, -16.2000, -17.6000,
]

# Waymo has no azimuth rotation offsets (uniform azimuth distribution)
WAYMO_TOP_AZ_OFFSET_DEG = [0.0] * 64

# Waymo lidar parameters
WAYMO_TOP_ROWS = 64
WAYMO_TOP_FOV_MIN_DEG = -17.6
WAYMO_TOP_FOV_MAX_DEG = 2.4

HDL64E_ROT_DEG = [
    -4.98, -2.91, 2.78, 5.0, -0.69, 1.58, -1.5, 0.79, 3.58, 5.81, 2.8, 5.02, -5.02,
    -2.76, -5.78, -3.52, -0.75, 1.51, -1.51, 0.73, 3.49, 5.74, 2.74, 4.97, -5.04, -2.78,
    -5.83, -3.54, -0.79, 1.46, -1.59, 0.71, -8.02, -4.42, 4.42, 7.78, -1.0, 2.49, -2.27,
    1.22, 5.84, 9.32, 4.71, 8.32, -7.75, -4.33, -9.19, -5.57, -1.03, 2.33, -2.27, 1.18,
    5.61, 8.97, 4.49, 7.92, -7.51, -4.31, -8.82, -5.42, -1.0, 2.3, -2.28, 1.07,
]

# Pandar128 params from gsplat tests (radians converted to degrees).
PANDAR128_ROWS = 128
PANDAR128_COLS = 3600
PANDAR128_ELEV_START_DEG = math.degrees(0.25195573)
PANDAR128_ELEV_END_DEG = math.degrees(-0.41325906)
PANDAR128_AZ_START_DEG = math.degrees(3.0847472798897)
PANDAR128_AZ_END_DEG = math.degrees(-3.196692698037892)

# PandaSet / Pandar64 distribution from the devkit tutorial CSV.
PANDAR64_VERT_DEG = [
    14.87, 11.02, 8.047, 5.045, 3.028, 2.016, 1.848, 1.676,
    1.51, 1.339, 1.172, 1.001, 0.834, 0.663, 0.496, 0.325,
    0.157, -0.012, -0.181, -0.349, -0.52, -0.687, -0.857, -1.025,
    -1.196, -1.363, -1.534, -1.7, -1.872, -2.04, -2.21, -2.377,
    -2.548, -2.712, -2.885, -3.052, -3.222, -3.387, -3.56, -3.724,
    -3.896, -4.062, -4.233, -4.397, -4.57, -4.732, -4.904, -5.069,
    -5.241, -5.403, -5.577, -5.738, -5.91, -6.073, -7.075, -8.071,
    -9.072, -9.897, -11.044, -12.018, -12.986, -13.942, -18.901, -24.909,
]
PANDAR64_ROT_DEG = [
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, 1.042, 3.125,
    5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208,
    -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
    1.042, 3.125, 5.208, -5.208, -3.125, -1.042, 1.042, 3.125,
    5.208, -5.208, -3.125, -1.042, 1.042, 3.125, 5.208, -5.208,
    -3.125, -1.042, 1.042, 3.125, 5.208, -5.208, -3.125, -1.042,
    1.042, 3.125, 5.208, -5.208, -3.125, -1.042, -1.042, -1.042,
    -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042, -1.042,
]


def _parse_dbxml(dbxml_path: str | Path) -> tuple[list[float], list[float]]:
    path = Path(dbxml_path)
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    vert = [float(e.text) for e in root.iter("vertCorrection_")]
    rot = [float(e.text) for e in root.iter("rotCorrection_")]
    if len(vert) != 64 or len(rot) != 64:
        raise ValueError(f"Expected 64 vert/rot corrections, got {len(vert)} / {len(rot)} in {path}")
    return vert, rot


def _move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.to(device=device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_move_to_device(v, device) for v in obj]
        return type(obj)(out)
    return obj


def _coeffs_cache_key(
    *,
    sensor: str,
    width: int,
    height: int,
    row_elevations_deg: list[float],
    row_azimuth_offsets_deg: list[float],
    az_start: float,
    az_end: float,
    spinning_frequency_hz: float,
    direction: str,
    resolution_factor: float,
) -> str:
    payload = dict(
        sensor=sensor,
        width=width,
        height=height,
        row_elevations_deg=[float(v) for v in row_elevations_deg],
        row_azimuth_offsets_deg=[float(v) for v in row_azimuth_offsets_deg],
        az_start=float(az_start),
        az_end=float(az_end),
        spinning_frequency_hz=float(spinning_frequency_hz),
        direction=str(direction),
        resolution_factor=float(resolution_factor),
        version=1,
    )
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def build_gsplat_lidar_coeffs(
    config: Any,
    width: int,
    height: int,
    device: torch.device,
) -> Any:
    use_cpu = False
    resolution_factor = 4.0
    resolution_factor_int = max(1, int(round(resolution_factor)))
    use_cache = False
    cache_dir = Path(".cache/lidar_coeffs")
    global gsplat, SpinningDirection, GSPLAT_IMPORT_ERROR
    if gsplat is None:
        try:
            import gsplat as _gsplat  # type: ignore
            from gsplat.cuda._wrapper import SpinningDirection as _SpinningDirection  # type: ignore
        except Exception as exc:  # pragma: no cover
            GSPLAT_IMPORT_ERROR = exc
            raise ImportError("gsplat is required for gsplat_lidar mode.") from GSPLAT_IMPORT_ERROR
        gsplat = _gsplat
        SpinningDirection = _SpinningDirection

    sensor = str(getattr(config, "lidar_sensor", "hdl64e")).lower()
    row_elevations_deg = getattr(config, "lidar_row_elevations_deg", None)
    if row_elevations_deg is None:
        row_elevations_deg = getattr(config, "lidar_vertical_angles_deg", None)
    row_azimuth_offsets_deg = getattr(config, "lidar_row_azimuth_offsets_deg", None)
    angle_mode = str(getattr(config, "lidar_angle_mode", "fixed")).lower()
    if row_elevations_deg is None:
        if angle_mode == "fixed":
            fov_max = float(getattr(config, "lidar_vertical_fov_max_deg", 4.0))
            fov_min = float(getattr(config, "lidar_vertical_fov_min_deg", -24.0))
            row_elevations_deg = torch.linspace(fov_max, fov_min, int(height)).tolist()
            row_azimuth_offsets_deg = [0.0] * int(height)
        elif sensor == "hdl64e":
            row_elevations_deg = HDL64E_VERT_DEG
            row_azimuth_offsets_deg = HDL64E_ROT_DEG
        elif sensor == "waymo" or sensor == "waymo_top":
            row_elevations_deg = WAYMO_TOP_VERT_DEG
            row_azimuth_offsets_deg = WAYMO_TOP_AZ_OFFSET_DEG
            if int(height) != WAYMO_TOP_ROWS:
                raise ValueError(
                    f"waymo_top expects lidar_height={WAYMO_TOP_ROWS}, got {height}"
                )
        elif sensor == "pandar128":
            row_elevations_deg = torch.linspace(
                PANDAR128_ELEV_START_DEG, PANDAR128_ELEV_END_DEG, PANDAR128_ROWS
            ).tolist()
            row_azimuth_offsets_deg = [0.0] * PANDAR128_ROWS
            if int(width) != PANDAR128_COLS or int(height) != PANDAR128_ROWS:
                raise ValueError(
                    f"pandar128 expects lidar_width={PANDAR128_COLS} and lidar_height={PANDAR128_ROWS}, "
                    f"got {width}x{height}"
                )
        elif sensor == "pandar64":
            row_elevations_deg = PANDAR64_VERT_DEG
            row_azimuth_offsets_deg = PANDAR64_ROT_DEG
            if int(height) != 64:
                raise ValueError(f"pandar64 expects lidar_height=64, got {height}")
        else:
            raise ValueError(
                "lidar_row_elevations_deg is required for custom sensors."
            )

    if row_azimuth_offsets_deg is None:
        if sensor == "hdl64e" and angle_mode == "fitted":
            row_azimuth_offsets_deg = HDL64E_ROT_DEG
        else:
            row_azimuth_offsets_deg = [0.0] * len(row_elevations_deg)
    if len(row_elevations_deg) != height:
        raise ValueError(
            f"lidar_row_elevations_deg length {len(row_elevations_deg)} does not match lidar_height {height}"
        )
    if len(row_azimuth_offsets_deg) != height:
        raise ValueError(
            f"lidar_row_azimuth_offsets_deg length {len(row_azimuth_offsets_deg)} does not match lidar_height {height}"
        )

    spinning_frequency_hz = float(getattr(config, "lidar_spinning_frequency_hz", 10.0))
    direction = str(getattr(config, "lidar_spinning_direction", "clockwise")).lower()
    if direction not in ("clockwise", "counterclockwise"):
        raise ValueError("lidar_spinning_direction must be 'clockwise' or 'counterclockwise'")

    if sensor == "pandar128":
        az_start = PANDAR128_AZ_START_DEG
        az_end = PANDAR128_AZ_END_DEG
    else:
        az_start = float(getattr(config, "lidar_column_azimuth_start_deg", 180.0))
        az_end = float(getattr(config, "lidar_column_azimuth_end_deg", -180.0))

    row_elev = torch.tensor([math.radians(v) for v in row_elevations_deg], device=device, dtype=torch.float32)
    row_az_off = torch.tensor([math.radians(v) for v in row_azimuth_offsets_deg], device=device, dtype=torch.float32)
    # Enforce descending order in cw-relative angle (gsplat requirement).
    sort_idx = torch.argsort(row_elev, descending=True)
    row_elev = row_elev[sort_idx]
    row_az_off = row_az_off[sort_idx]
    n_columns = int(width)
    start_rad = math.radians(az_start)
    end_rad = math.radians(az_end)
    # Avoid wrap-around by keeping end slightly inside the span.
    if abs((az_start - az_end) % 360.0) < 1.0e-3:
        # Full 360 scan: cover almost 2*pi to avoid wrap-around duplication.
        delta = 2.0 * math.pi / max(n_columns, 1)
        span = 2.0 * math.pi - delta
        end_rad = start_rad - span if direction == "clockwise" else start_rad + span

    column_az = torch.linspace(
        start_rad,
        end_rad,
        n_columns,
        device=device,
        dtype=torch.float32,
    )
    params = dict(
        spinning_direction=SpinningDirection.CLOCKWISE if direction == "clockwise" else SpinningDirection.COUNTER_CLOCKWISE,
        spinning_frequency_hz=spinning_frequency_hz,
        row_elevations_rad=row_elev,
        column_azimuths_rad=column_az,
        row_azimuth_offsets_rad=row_az_off,
    )

    cache_key = _coeffs_cache_key(
        sensor=sensor,
        width=int(width),
        height=int(height),
        row_elevations_deg=row_elevations_deg,
        row_azimuth_offsets_deg=row_azimuth_offsets_deg,
        az_start=az_start,
        az_end=az_end,
        spinning_frequency_hz=spinning_frequency_hz,
        direction=direction,
        resolution_factor=resolution_factor,
    )
    cache_file = cache_dir / f"{cache_key}.pt"
    cache_meta_file = cache_dir / f"{cache_key}.json"
    if use_cache and cache_file.exists():
        payload = torch.load(cache_file, map_location="cpu")
        params["angles_to_columns_map"] = _move_to_device(payload["angles_to_columns_map"], device)
        params["tiling"] = _move_to_device(payload["tiling"], device)
        return gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(**params)

    base = gsplat.RowOffsetStructuredSpinningLidarModelParameters(**params)
    if use_cpu:
        base_cpu = gsplat.RowOffsetStructuredSpinningLidarModelParameters(
            spinning_direction=params["spinning_direction"],
            spinning_frequency_hz=params["spinning_frequency_hz"],
            row_elevations_rad=row_elev.cpu(),
            column_azimuths_rad=column_az.cpu(),
            row_azimuth_offsets_rad=row_az_off.cpu(),
        )
        angles_map = gsplat.compute_lidar_angles_to_columns_map(base_cpu, resolution_factor=resolution_factor_int)
        tiling = gsplat.compute_lidar_tiling(
            base_cpu,
            n_bins_elevation=16,
            max_pts_per_tile=64,
            resolution_elevation=1600,
            densification_factor_azimuth=8,
        )
        params["angles_to_columns_map"] = _move_to_device(angles_map, device)
        params["tiling"] = _move_to_device(tiling, device)
        if use_cache:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"angles_to_columns_map": angles_map, "tiling": tiling},
                    cache_file,
                )
                cache_meta_file.write_text(
                    json.dumps(
                        {
                            "sensor": sensor,
                            "width": int(width),
                            "height": int(height),
                            "az_start": float(az_start),
                            "az_end": float(az_end),
                            "spinning_frequency_hz": float(spinning_frequency_hz),
                            "direction": direction,
                            "resolution_factor": float(resolution_factor),
                            "cache_key": cache_key,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            except Exception as _e:
                pass
    else:
        params["angles_to_columns_map"] = gsplat.compute_lidar_angles_to_columns_map(
            base, resolution_factor=resolution_factor_int
        )
        params["tiling"] = gsplat.compute_lidar_tiling(
            base,
            n_bins_elevation=16,
            max_pts_per_tile=64,
            resolution_elevation=1600,
            densification_factor_azimuth=8,
        )

    out = gsplat.RowOffsetStructuredSpinningLidarModelParametersExt(**params)
    return out
