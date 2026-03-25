from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .lidar_projection import elevation_to_row


@dataclass
class FrameSample:
    rgb: torch.Tensor
    lidar_depth: torch.Tensor
    # Optional raw LiDAR points in LiDAR frame, shape [N,3]
    lidar_points: torch.Tensor | None = None
    camera_to_world: torch.Tensor = None  # type: ignore[assignment]
    lidar_to_world: torch.Tensor = None  # type: ignore[assignment]
    intrinsics: torch.Tensor = None  # type: ignore[assignment]
    rgb_width: int = 0
    rgb_height: int = 0
    lidar_width: int = 0
    lidar_height: int = 0
    frame_id: str = ""
    # Dynamic mask: 0=static, >0=dynamic. Used to filter dynamic regions in loss.
    dynamic_mask: torch.Tensor | None = None


class MultiSensorDataset(Dataset[FrameSample]):
    def __init__(
        self,
        manifest_path: str,
        rgb_tensor_key: str | None = None,
        lidar_depth_key: str | None = None,
    ) -> None:
        self.root = Path(manifest_path).resolve().parent
        self.samples = [
            Path(line.strip())
            for line in Path(manifest_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.rgb_tensor_key = rgb_tensor_key
        self.lidar_depth_key = lidar_depth_key

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> FrameSample:
        meta_path = self._resolve_path(self.samples[index])
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        rgb = self._load_tensor(self._resolve_path(Path(metadata["rgb_path"])), self.rgb_tensor_key)
        lidar_depth = self._load_tensor(self._resolve_path(Path(metadata["lidar_depth_path"])), self.lidar_depth_key)
        camera_to_world = self._to_homogeneous_pose(metadata["camera_c2w"])
        lidar_to_world = self._to_homogeneous_pose(metadata["lidar_c2w"])
        intrinsics = torch.tensor(metadata["intrinsics"], dtype=torch.float32)

        rgb_height = int(metadata.get("rgb_height", rgb.shape[-2]))
        rgb_width = int(metadata.get("rgb_width", rgb.shape[-1]))
        lidar_height = int(metadata.get("lidar_height", lidar_depth.shape[-2]))
        lidar_width = int(metadata.get("lidar_width", lidar_depth.shape[-1]))

        return FrameSample(
            rgb=rgb,
            lidar_depth=lidar_depth,
            lidar_points=None,
            camera_to_world=camera_to_world,
            lidar_to_world=lidar_to_world,
            intrinsics=intrinsics,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            lidar_width=lidar_width,
            lidar_height=lidar_height,
            frame_id=metadata.get("frame_id", meta_path.stem),
        )

    def _resolve_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.root / path).resolve()

    def _load_tensor(self, path: Path, tensor_key: str | None) -> torch.Tensor:
        if path.suffix == ".npy":
            payload: Any = np.load(path)
        else:
            payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            if tensor_key is None:
                raise KeyError(f"tensor_key must be set when loading dict payload: {path}")
            payload = payload[tensor_key]
        tensor = torch.as_tensor(payload, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"Expected [C, H, W] or [H, W], got shape {tuple(tensor.shape)} from {path}")
        if tensor.shape[0] == 3 and tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    @staticmethod
    def _to_homogeneous_pose(raw_pose: list[list[float]]) -> torch.Tensor:
        pose = torch.tensor(raw_pose, dtype=torch.float32)
        if pose.shape == (3, 4):
            pose = torch.cat([pose, torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)], dim=0)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be [3, 4] or [4, 4], got shape {tuple(pose.shape)}")
        return pose


class KittiRDataset(Dataset[FrameSample]):
    """Dataset loader compatible with HiGS-Calib KITTI_R layout.

    Produces FrameSample with the same keys/semantics as MultiSensorDataset.
    """

    def __init__(
        self,
        source_path: str,
        data_seq: str,
        cam_id: str = "02",
        start_index: int = 0,
        data_type: str = "t",
        segment_length: int = 50,
        lidar_width: int = 2048,
        lidar_height: int = 64,
        max_range: float = 50.0,
        near_plane: float = 0.1,
        far_plane: float = 120.0,
        lidar_vertical_fov_min_deg: float = -15.0,
        lidar_vertical_fov_max_deg: float = 15.0,
        lidar_vertical_angles_deg: list[float] | None = None,
        lidar_vertical_angle_offset_deg: float = 0.0,
        lidar_angle_mode: str = "fixed",
        lidar_dbxml_path: str | None = None,
    ) -> None:
        self.source_root = Path(source_path)
        # Allow passing either the dataset root or a parent directory containing 'kitti-calibration'.
        if (self.source_root / "kitti-calibration").exists() and (self.source_root / "calibs").exists() is False:
            self.source_root = self.source_root / "kitti-calibration"
        self.data_seq = str(data_seq)
        self.cam_id = cam_id
        self.start_index = int(start_index)
        self.data_type = data_type
        self.segment_length = int(segment_length)
        self.lidar_width = int(lidar_width)
        self.lidar_height = int(lidar_height)
        self.max_range = float(max_range)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.lidar_vertical_fov_min_deg = float(lidar_vertical_fov_min_deg)
        self.lidar_vertical_fov_max_deg = float(lidar_vertical_fov_max_deg)
        self.lidar_vertical_angle_offset_deg = float(lidar_vertical_angle_offset_deg)
        # Defer auto-estimation until scene_root is available.
        self.lidar_vertical_angles_deg = lidar_vertical_angles_deg
        self.lidar_angle_mode = str(lidar_angle_mode)
        self.lidar_dbxml_path = lidar_dbxml_path

        scene_num = int(self.data_seq.lstrip("0") or "0")
        self.scene_name = f"{scene_num}-{self.start_index}-{self.data_type}"
        self.scene_root = self.source_root / self.scene_name

        calib_path = self.source_root / "calibs" / f"{int(self.data_seq):02d}.txt"
        self._intrinsics = self._load_intrinsics(
            calib_path=calib_path,
            cam_id=self.cam_id,
        )
        self._T_cam_velo = self._load_lidar_to_camera(
            scene_root=self.scene_root,
            calib_path=calib_path,
        )
        # Match HiGS-Calib: poses are loaded from scene_root/LiDAR_poses.txt and normalized by inv(poses[0]) @ poses.
        poses_path = self.scene_root / "LiDAR_poses.txt"
        has_per_scene_poses = poses_path.exists()
        if not has_per_scene_poses:
            # Fallback for datasets that store poses centrally.
            poses_path = self.source_root / "poses" / f"{int(self.data_seq):02d}.txt"
        self._lidar_poses = self._load_lidar_poses(poses_path)

        if len(self._lidar_poses) < self.segment_length:
            raise ValueError(
                f"Not enough poses in {poses_path}: {len(self._lidar_poses)} < {self.segment_length}"
            )

        pose0_inv = np.linalg.inv(self._lidar_poses[0])
        self._lidar_poses = [pose0_inv @ p for p in self._lidar_poses]

        # For per-scene pre-segmented data (LiDAR_poses.txt in scene directory),
        # start_index is encoded in the scene name and should be ignored.
        # The scene already contains the correct segment of frames (00.png, 01.png, etc.)
        if has_per_scene_poses:
            # Pre-segmented scene: start_index is implicit from scene directory name.
            # Reset to 0 since poses are already relative to this segment.
            self._start_index_offset = self.start_index
            self.start_index = 0
        else:
            # Central poses file: apply start_index normally.
            if not (0 <= self.start_index < len(self._lidar_poses)):
                raise ValueError(f"start_index {self.start_index} out of range for {len(self._lidar_poses)} poses")
            self._start_index_offset = 0

        shift = self._lidar_poses[self.start_index][:3, 3].copy()
        for i in range(len(self._lidar_poses)):
            self._lidar_poses[i] = self._lidar_poses[i].copy()
            self._lidar_poses[i][:3, 3] -= shift

        # If an empty list is provided, auto-estimate angles from data.
        if self.lidar_angle_mode == "fitted":
            if self.lidar_dbxml_path:
                self.lidar_vertical_angles_deg = self._load_dbxml_vertical_angles(self.lidar_dbxml_path)
            else:
                self.lidar_vertical_angles_deg = self._estimate_vertical_angles(
                    frames=min(20, self.segment_length),
                    near_plane=self.near_plane,
                    far_plane=self.far_plane,
                )
        self._apply_vertical_angle_offset()

    def __len__(self) -> int:
        return self.segment_length

    def __getitem__(self, index: int) -> FrameSample:
        # HiGS-Calib KITTI_R scenes store frames as local indices 00..(segment_length-1)
        # regardless of the global start_index encoded in the scene_name.
        local_id = int(index)
        rgb_path = self.scene_root / f"{local_id:02d}.png"
        lidar_path = self.scene_root / f"{local_id:02d}.txt"

        rgb = self._load_png_as_tensor(rgb_path)
        points = self._load_pointcloud_txt(lidar_path, max_range=self.max_range)
        lidar_points = torch.from_numpy(points.astype(np.float32))
        lidar_depth = self._points_to_lidar_depth(
            points,
            width=self.lidar_width,
            height=self.lidar_height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            vertical_fov_min_deg=self.lidar_vertical_fov_min_deg,
            vertical_fov_max_deg=self.lidar_vertical_fov_max_deg,
            vertical_angles_deg=self.lidar_vertical_angles_deg,
        )

        lidar_to_world = torch.tensor(self._lidar_poses[local_id], dtype=torch.float32)

        # Match HiGS-Calib SensorTrajectories.get_camera_pose_gt() exactly.
        # Here _T_cam_velo is T_cl (camera-from-lidar), so camera pose in world is:
        #   R_wc = R_wl @ R_cl^T
        #   t_wc = -R_wc @ t_cl + t_wl
        T_wl = self._lidar_poses[local_id]
        R_wl = T_wl[:3, :3]
        t_wl = T_wl[:3, 3]
        R_cl = self._T_cam_velo[:3, :3]
        t_cl = self._T_cam_velo[:3, 3]
        R_wc = R_wl @ R_cl.T
        t_wc = -(R_wc @ t_cl) + t_wl
        T_wc = np.eye(4, dtype=np.float32)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = t_wc
        camera_to_world = torch.tensor(T_wc, dtype=torch.float32)

        intrinsics = torch.tensor(self._intrinsics, dtype=torch.float32)

        rgb_height = int(rgb.shape[-2])
        rgb_width = int(rgb.shape[-1])

        return FrameSample(
            rgb=rgb,
            lidar_depth=lidar_depth,
            lidar_points=lidar_points,
            camera_to_world=camera_to_world,
            lidar_to_world=lidar_to_world,
            intrinsics=intrinsics,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            lidar_width=self.lidar_width,
            lidar_height=self.lidar_height,
            frame_id=f"{self.scene_name}_{local_id:02d}",
        )

    @staticmethod
    def _load_intrinsics(calib_path: Path, cam_id: str) -> np.ndarray:
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")
        lines = calib_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            raise ValueError(f"Empty calibration file: {calib_path}")

        # Prefer P{cam_id} if present.
        # NOTE: cam_id is a camera index in the multi-camera rig (e.g., 2), not an image name.
        target = f"P{int(cam_id):d}" if cam_id.isdigit() else f"P{cam_id}"
        floats: list[float] = []

        for line in lines:
            parts = line.replace(":", " ").split()
            if not parts:
                continue
            if parts[0] == target:
                vals = [p for p in parts[1:] if _is_float(p)]
                if len(vals) >= 12:
                    floats = [float(v) for v in vals[:12]]
                    break

        if not floats:
            for line in lines:
                parts = line.replace(":", " ").split()
                if not parts:
                    continue
                if parts[0].startswith("P") and any(ch.isdigit() for ch in parts[0]):
                    vals = [p for p in parts[1:] if _is_float(p)]
                    if len(vals) >= 12:
                        floats = [float(v) for v in vals[:12]]
                        break

        if not floats:
            for token in " ".join(lines).replace(":", " ").split():
                if _is_float(token):
                    floats.append(float(token))
                if len(floats) >= 12:
                    break

        if len(floats) < 12:
            raise ValueError(f"Could not parse 3x4 P matrix from {calib_path}")

        P = np.array(floats[:12], dtype=np.float32).reshape(3, 4)
        return P[:3, :3].copy()

    @staticmethod
    def _load_lidar_to_camera(scene_root: Path, calib_path: Path) -> np.ndarray:
        """Load T_cam_velo (camera-from-lidar).

        Priority:
        1) HiGS-Calib per-scene LiDAR-to-camera.json (if present)
        2) KITTI-style calib 'Tr:' line in calibs/{seq}.txt
        """
        path = scene_root / "LiDAR-to-camera.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "correct" not in payload:
                raise KeyError(f"Expected key 'correct' in {path}")
            T = np.array(payload["correct"], dtype=np.float32)
            if T.shape != (4, 4):
                raise ValueError(f"Expected 4x4 matrix in {path}['correct'], got {T.shape}")
            return T

        # Fallback: parse KITTI-style calib 'Tr:' line (3x4) as T_cam_velo.
        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found (needed for Tr): {calib_path}")
        lines = calib_path.read_text(encoding="utf-8").strip().splitlines()
        tr_floats: list[float] = []
        for line in lines:
            parts = line.replace(":", " ").split()
            if not parts:
                continue
            if parts[0] == "Tr":
                vals = [p for p in parts[1:] if _is_float(p)]
                if len(vals) >= 12:
                    tr_floats = [float(v) for v in vals[:12]]
                    break
        if len(tr_floats) < 12:
            raise FileNotFoundError(f"Could not find 'Tr:' in {calib_path} and no LiDAR-to-camera.json in {scene_root}")

        Tr = np.array(tr_floats, dtype=np.float32).reshape(3, 4)
        T = np.eye(4, dtype=np.float32)
        T[:3, :4] = Tr
        return T

    @staticmethod
    def _load_lidar_poses(path: Path) -> list[np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"LiDAR_poses.txt not found: {path}")
        poses: list[np.ndarray] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            values = [float(v) for v in line.split()]
            if len(values) == 12:
                mat = np.array(values, dtype=np.float32).reshape(3, 4)
                mat = np.vstack([mat, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
            elif len(values) == 16:
                mat = np.array(values, dtype=np.float32).reshape(4, 4)
            else:
                raise ValueError(f"Pose line must have 12 or 16 floats, got {len(values)} in {path}")
            poses.append(mat)
        if not poses:
            raise ValueError(f"No poses found in {path}")
        return poses

    @staticmethod
    def _load_png_as_tensor(path: Path) -> torch.Tensor:
        # Use PIL if available; otherwise error with actionable message.
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise ImportError("KittiRDataset requires pillow to read .png (pip install pillow).") from exc

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

    @staticmethod
    def _load_dbxml_vertical_angles(dbxml_path: str) -> list[float]:
        root = ET.fromstring(Path(dbxml_path).read_text(encoding="utf-8"))
        vert = [float(e.text) for e in root.iter("vertCorrection_")]
        if len(vert) != 64:
            raise ValueError(f"Expected 64 vertCorrection_ entries in dbxml, got {len(vert)} from {dbxml_path}")
        return sorted(vert, reverse=True)

    def _apply_vertical_angle_offset(self) -> None:
        if self.lidar_vertical_angles_deg is None or not len(self.lidar_vertical_angles_deg):
            return
        offset = float(self.lidar_vertical_angle_offset_deg)
        if abs(offset) <= 1.0e-9:
            return
        self.lidar_vertical_angles_deg = [float(a) + offset for a in self.lidar_vertical_angles_deg]

    @staticmethod
    def _load_pointcloud_txt(path: Path, max_range: float) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Point cloud txt not found: {path}")
        pts = np.loadtxt(path, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] < 3:
            raise ValueError(f"Point cloud must have at least 3 columns (x,y,z): {path}")
        xyz = pts[:, :3]
        r = np.linalg.norm(xyz, axis=1)
        return xyz[r < float(max_range)]

    @staticmethod
    def _points_to_lidar_depth(
        points_lidar: np.ndarray,
        width: int,
        height: int,
        near_plane: float,
        far_plane: float,
        vertical_fov_min_deg: float,
        vertical_fov_max_deg: float,
        vertical_angles_deg: list[float] | None = None,
    ) -> torch.Tensor:
        # Match GaussianSceneModel.render_lidar spherical mapping.
        if points_lidar.size == 0:
            return torch.zeros((1, height, width), dtype=torch.float32)

        x = points_lidar[:, 0]
        y = points_lidar[:, 1]
        z = points_lidar[:, 2]

        # KITTI Velodyne convention: x forward, y left, z up.
        # Therefore elevation is measured against the horizontal plane (x-y).
        xy_norm = np.sqrt(x * x + y * y)
        xy_norm = np.clip(xy_norm, 1.0e-6, None)
        ranges = np.sqrt(x * x + y * y + z * z)
        ranges = np.clip(ranges, 1.0e-6, None)

        # KITTI Velodyne convention: azimuth is the yaw around the vertical axis (z).
        # This affects horizontal indexing and must match the model-side renderer.
        azimuth = np.arctan2(y, x)
        elevation = np.arctan2(z, xy_norm)

        min_el = math.radians(vertical_fov_min_deg)
        max_el = math.radians(vertical_fov_max_deg)
        valid = (ranges >= near_plane) & (ranges <= far_plane) & (elevation >= min_el) & (elevation <= max_el)
        if not np.any(valid):
            return torch.zeros((1, height, width), dtype=torch.float32)

        azimuth = azimuth[valid]
        elevation = elevation[valid]
        ranges = ranges[valid]

        u = (azimuth + math.pi) / (2.0 * math.pi) * float(width)
        v = elevation_to_row(
            elevation=elevation,
            height=height,
            vertical_fov_min_deg=vertical_fov_min_deg,
            vertical_fov_max_deg=vertical_fov_max_deg,
            vertical_angles_deg=vertical_angles_deg,
        )

        # Hard depth (first return) for supervision: per-pixel nearest range.
        # Note: this is intentionally different from the model-side soft aggregation depth.
        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        ui = np.mod(ui, width)
        vi = np.clip(vi, 0, height - 1)

        depth = np.zeros((height, width), dtype=np.float32)
        flat_idx = vi * width + ui
        order = np.argsort(ranges)  # nearest first
        flat_idx = flat_idx[order]
        ranges_sorted = ranges[order]

        seen = np.zeros(height * width, dtype=bool)
        depth_flat = depth.reshape(-1)
        for idx, r in zip(flat_idx, ranges_sorted, strict=False):
            if not seen[idx]:
                depth_flat[idx] = r
                seen[idx] = True

        return torch.from_numpy(depth).unsqueeze(0)

    def _estimate_vertical_angles(
        self,
        frames: int = 20,
        near_plane: float = 0.1,
        far_plane: float = 120.0,
    ) -> list[float]:
        """Estimate per-ring vertical angles (deg) from point clouds.

        Uses elevation quantiles to produce `lidar_height` angles.
        """
        elevations: list[np.ndarray] = []
        for i in range(frames):
            lidar_path = self.scene_root / f"{i:02d}.txt"
            if not lidar_path.exists():
                continue
            pts = self._load_pointcloud_txt(lidar_path, max_range=self.max_range)
            if pts.size == 0:
                continue
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            xy = np.sqrt(x * x + y * y)
            xy = np.clip(xy, 1.0e-6, None)
            r = np.sqrt(x * x + y * y + z * z)
            valid = (r >= near_plane) & (r <= far_plane)
            if not np.any(valid):
                continue
            el = np.arctan2(z[valid], xy[valid])
            elevations.append(el)
        if not elevations:
            return []
        el_all = np.concatenate(elevations, axis=0)
        # Use quantiles to approximate ring angles (median per slice).
        qs = (np.arange(self.lidar_height, dtype=np.float32) + 0.5) / float(self.lidar_height)
        angles = np.quantile(el_all, qs)
        angles = np.degrees(angles).tolist()
        return sorted(angles, reverse=True)


def multi_sensor_collate_fn(batch: list[FrameSample]) -> dict[str, Any]:
    if len(batch) != 1:
        raise ValueError("This minimal framework currently expects batch_size=1 for variable-sized sensors.")
    sample = batch[0]
    out: dict[str, Any] = {
        "rgb": sample.rgb,
        "lidar_depth": sample.lidar_depth,
        "camera_to_world": sample.camera_to_world,
        "lidar_to_world": sample.lidar_to_world,
        "intrinsics": sample.intrinsics,
        "rgb_width": sample.rgb_width,
        "rgb_height": sample.rgb_height,
        "lidar_width": sample.lidar_width,
        "lidar_height": sample.lidar_height,
        "frame_id": sample.frame_id,
    }
    if sample.lidar_points is not None:
        out["lidar_points"] = sample.lidar_points
    if sample.dynamic_mask is not None:
        out["dynamic_mask"] = sample.dynamic_mask
    return out


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except Exception:
        return False
