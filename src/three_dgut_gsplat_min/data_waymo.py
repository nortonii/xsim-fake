from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FrameSample
from .lidar_projection import elevation_to_row


class WaymoDataset(Dataset[FrameSample]):
    """Waymo scene loader with KITTI_R-like output semantics.

    Expected scene layout (example):
      /root/open_dataset/waymo/002/
        images/000000_0.png
        intrinsics/0.txt
        extrinsics/0.txt
        ego_pose/000000.txt
        ego_pose/000000_0.txt
        pointcloud.npz
    """

    def __init__(
        self,
        source_path: str,
        scene_id: str,
        cam_id: str = "0",
        start_index: int = 0,
        segment_length: int | None = None,
        lidar_width: int = 2048,
        lidar_height: int = 64,
        max_range: float = 80.0,
        near_plane: float = 0.1,
        far_plane: float = 120.0,
        lidar_vertical_fov_min_deg: float = -17.6,
        lidar_vertical_fov_max_deg: float = 2.4,
        lidar_vertical_angles_deg: list[float] | None = None,
    ) -> None:
        self.source_root = Path(source_path)
        self.scene_id = str(scene_id)
        self.cam_id = str(int(cam_id))
        self.start_index = int(start_index)
        self.scene_root = self.source_root / self.scene_id

        if not self.scene_root.exists():
            raise FileNotFoundError(f"Waymo scene root not found: {self.scene_root}")

        self.images_dir = self.scene_root / "images"
        self.intrinsics_dir = self.scene_root / "intrinsics"
        self.extrinsics_dir = self.scene_root / "extrinsics"
        self.ego_pose_dir = self.scene_root / "ego_pose"
        self.pointcloud_path = self.scene_root / "pointcloud.npz"
        self.dynamic_mask_dir = self.scene_root / "dynamic_mask"

        self.lidar_width = int(lidar_width)
        self.lidar_height = int(lidar_height)
        self.max_range = float(max_range)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.lidar_vertical_fov_min_deg = float(lidar_vertical_fov_min_deg)
        self.lidar_vertical_fov_max_deg = float(lidar_vertical_fov_max_deg)
        self.lidar_vertical_angles_deg = lidar_vertical_angles_deg

        self._intrinsics = self._load_intrinsics(self.intrinsics_dir / f"{self.cam_id}.txt")
        self._T_vehicle_cam = self._load_pose_txt(self.extrinsics_dir / f"{self.cam_id}.txt")

        self._frame_ids = self._collect_frame_ids_for_camera(self.images_dir, self.cam_id)
        if not self._frame_ids:
            raise ValueError(f"No frames found for cam_id={self.cam_id} under {self.images_dir}")

        self._pointcloud = self._load_pointcloud_dict(self.pointcloud_path)

        if self.start_index < 0 or self.start_index >= len(self._frame_ids):
            raise ValueError(f"start_index {self.start_index} out of range [0, {len(self._frame_ids) - 1}]")

        available = len(self._frame_ids) - self.start_index
        if segment_length is None:
            self.segment_length = available
        else:
            self.segment_length = int(segment_length)
            if self.segment_length <= 0:
                raise ValueError("segment_length must be > 0")
            if self.segment_length > available:
                raise ValueError(
                    f"segment_length {self.segment_length} exceeds available frames ({available}) "
                    f"from start_index={self.start_index}"
                )

    def __len__(self) -> int:
        return self.segment_length

    def __getitem__(self, index: int) -> FrameSample:
        frame_id = int(self._frame_ids[self.start_index + int(index)])

        rgb_path = self.images_dir / f"{frame_id:06d}_{self.cam_id}.png"
        rgb = self._load_png_as_tensor(rgb_path)

        # Vehicle pose in world; used as lidar pose proxy.
        lidar_to_world_np = self._load_pose_txt(self.ego_pose_dir / f"{frame_id:06d}.txt")
        lidar_to_world = torch.tensor(lidar_to_world_np, dtype=torch.float32)

        # Use calibrated extrinsics to build camera pose from vehicle pose.
        # This keeps the lidar/camera transform physically consistent for projection.
        camera_to_world_np = lidar_to_world_np @ self._T_vehicle_cam
        camera_to_world = torch.tensor(camera_to_world_np, dtype=torch.float32)

        # Point cloud indexed by frame id in pointcloud.npz
        points = np.asarray(self._pointcloud[frame_id], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected Nx3 point cloud for frame {frame_id}, got shape {points.shape}")
        points = points[:, :3]
        r = np.linalg.norm(points, axis=1)
        valid = (r >= self.near_plane) & (r <= self.far_plane) & (r < self.max_range)
        points = points[valid]
        lidar_points = torch.from_numpy(points.astype(np.float32))

        lidar_depth = self._points_to_lidar_depth(
            points_lidar=points,
            width=self.lidar_width,
            height=self.lidar_height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            vertical_fov_min_deg=self.lidar_vertical_fov_min_deg,
            vertical_fov_max_deg=self.lidar_vertical_fov_max_deg,
            vertical_angles_deg=self.lidar_vertical_angles_deg,
        )

        intrinsics = torch.tensor(self._intrinsics, dtype=torch.float32)
        rgb_height = int(rgb.shape[-2])
        rgb_width = int(rgb.shape[-1])

        # Load dynamic mask if available
        dynamic_mask = self._load_dynamic_mask(frame_id, self.cam_id, self.dynamic_mask_dir)

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
            frame_id=f"{self.scene_id}_{frame_id:06d}_{self.cam_id}",
            dynamic_mask=dynamic_mask,
        )

    @staticmethod
    def _load_dynamic_mask(frame_id: int, cam_id: str, mask_dir: Path) -> torch.Tensor | None:
        """Load dynamic mask for a frame. Returns None if mask doesn't exist."""
        mask_path = mask_dir / f"{frame_id:06d}_{cam_id}.png"
        if not mask_path.exists():
            return None
        try:
            from PIL import Image  # type: ignore
            img = Image.open(mask_path)
            arr = np.asarray(img, dtype=np.uint8).copy()
            # Convert to binary mask: 0=static, 1=dynamic
            mask = (arr > 0).astype(np.float32)
            return torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        except Exception:
            return None

    @staticmethod
    def _collect_frame_ids_for_camera(images_dir: Path, cam_id: str) -> list[int]:
        frame_ids: list[int] = []
        suffix = f"_{cam_id}.png"
        for path in images_dir.glob(f"*{suffix}"):
            stem = path.stem
            token = stem.split("_")[0]
            if token.isdigit():
                frame_ids.append(int(token))
        frame_ids = sorted(set(frame_ids))
        return frame_ids

    @staticmethod
    def _load_pointcloud_dict(path: Path) -> dict[int, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"pointcloud.npz not found: {path}")
        payload = np.load(path, allow_pickle=True)
        if "pointcloud" not in payload.files:
            raise KeyError(f"Missing 'pointcloud' key in {path}. Found keys: {payload.files}")
        pointcloud = payload["pointcloud"].item()
        if not isinstance(pointcloud, dict):
            raise ValueError(f"Expected dict in {path}['pointcloud'], got {type(pointcloud)}")
        return pointcloud

    @staticmethod
    def _load_intrinsics(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {path}")
        vals = np.loadtxt(path, dtype=np.float32).reshape(-1)
        if vals.size < 4:
            raise ValueError(f"Expected >=4 values [fx, fy, cx, cy, ...], got {vals.size} in {path}")
        fx, fy, cx, cy = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return K

    @staticmethod
    def _load_pose_txt(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Pose file not found: {path}")
        pose = np.loadtxt(path, dtype=np.float32)
        if pose.shape == (3, 4):
            pose = np.vstack([pose, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)])
        if pose.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose in {path}, got shape {pose.shape}")
        return pose

    @staticmethod
    def _load_png_as_tensor(path: Path) -> torch.Tensor:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            raise ImportError("WaymoDataset requires pillow to read .png (pip install pillow).") from exc

        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8).copy()
        return torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

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
        if points_lidar.size == 0:
            return torch.zeros((1, height, width), dtype=torch.float32)

        x = points_lidar[:, 0]
        y = points_lidar[:, 1]
        z = points_lidar[:, 2]

        xy_norm = np.sqrt(x * x + y * y)
        xy_norm = np.clip(xy_norm, 1.0e-6, None)
        ranges = np.sqrt(x * x + y * y + z * z)
        ranges = np.clip(ranges, 1.0e-6, None)

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

        ui = np.floor(u).astype(np.int64)
        vi = np.floor(v).astype(np.int64)
        ui = np.mod(ui, width)
        vi = np.clip(vi, 0, height - 1)

        depth = np.zeros((height, width), dtype=np.float32)
        flat_idx = vi * width + ui
        order = np.argsort(ranges)
        flat_idx = flat_idx[order]
        ranges_sorted = ranges[order]

        seen = np.zeros(height * width, dtype=bool)
        depth_flat = depth.reshape(-1)
        for idx, r in zip(flat_idx, ranges_sorted, strict=False):
            if not seen[idx]:
                depth_flat[idx] = r
                seen[idx] = True

        return torch.from_numpy(depth).unsqueeze(0)
