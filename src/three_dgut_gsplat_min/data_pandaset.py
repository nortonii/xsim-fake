from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import FrameSample, KittiRDataset
from .lidar_models import PANDAR64_ROT_DEG, PANDAR64_VERT_DEG
from .lidar_projection import points_to_angle_table_depth


class PandaSetDataset(Dataset[FrameSample]):
    """PandaSet loader with KITTI/Waymo-like output semantics."""

    def __init__(
        self,
        source_path: str,
        sequence_id: str,
        camera_name: str = "front_camera",
        lidar_sensor_id: int = 0,
        start_index: int = 0,
        segment_length: int | None = None,
        lidar_width: int = 2048,
        lidar_height: int = 64,
        max_range: float = 80.0,
        near_plane: float = 0.1,
        far_plane: float = 120.0,
        lidar_vertical_fov_min_deg: float = -25.0,
        lidar_vertical_fov_max_deg: float = 20.0,
        lidar_vertical_angles_deg: list[float] | None = None,
        lidar_row_azimuth_offsets_deg: list[float] | None = None,
        lidar_vertical_angle_offset_deg: float = 0.0,
        lidar_angle_mode: str = "fitted",
    ) -> None:
        try:
            from pandaset import DataSet
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "PandaSetDataset requires the 'pandaset' package. "
                "Install it from submodules/pandaset-devkit/python first."
            ) from exc

        self.source_root = self._resolve_source_root(Path(source_path))
        self.sequence_id = str(sequence_id)
        self.camera_name = str(camera_name)
        self.lidar_sensor_id = int(lidar_sensor_id)
        self.start_index = int(start_index)
        self.lidar_width = int(lidar_width)
        self.lidar_height = int(lidar_height)
        self.max_range = float(max_range)
        self.near_plane = float(near_plane)
        self.far_plane = float(far_plane)
        self.lidar_vertical_fov_min_deg = float(lidar_vertical_fov_min_deg)
        self.lidar_vertical_fov_max_deg = float(lidar_vertical_fov_max_deg)
        self.lidar_vertical_angles_deg = lidar_vertical_angles_deg
        self.lidar_row_azimuth_offsets_deg = lidar_row_azimuth_offsets_deg
        self.lidar_vertical_angle_offset_deg = float(lidar_vertical_angle_offset_deg)
        self.lidar_angle_mode = str(lidar_angle_mode)
        self.scene_name = f"pandaset_{self.sequence_id}_{self.camera_name}"

        self._dataset = DataSet(str(self.source_root))
        sequences = set(self._dataset.sequences())
        if self.sequence_id not in sequences:
            available = ", ".join(sorted(sequences)[:10])
            raise ValueError(
                f"PandaSet sequence {self.sequence_id!r} not found under {self.source_root}. "
                f"Available examples: {available}"
            )

        self._sequence = self._dataset[self.sequence_id]
        self._sequence.load_lidar()
        self._sequence.lidar.set_sensor(self.lidar_sensor_id)
        if self.camera_name not in self._sequence.camera:
            raise ValueError(
                f"Unknown PandaSet camera {self.camera_name!r}. "
                f"Available cameras: {sorted(self._sequence.camera.keys())}"
            )
        self._sequence.camera[self.camera_name].load()

        self._camera = self._sequence.camera[self.camera_name]
        self._lidar = self._sequence.lidar
        self._intrinsics = self._camera_intrinsics_to_matrix(self._camera.intrinsics)

        n_frames = min(
            len(self._camera.data),
            len(self._camera.poses),
            len(self._lidar.data),
            len(self._lidar.poses),
        )
        if n_frames <= 0:
            raise ValueError(f"No synchronized PandaSet frames found for sequence {self.sequence_id}")

        if self.start_index < 0 or self.start_index >= n_frames:
            raise ValueError(f"start_index {self.start_index} out of range [0, {n_frames - 1}]")

        available = n_frames - self.start_index
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

        self._lidar_pose_mats = [self._pose_dict_to_matrix(p) for p in self._lidar.poses[:n_frames]]
        self._camera_pose_mats = [self._pose_dict_to_matrix(p) for p in self._camera.poses[:n_frames]]
        self._reference_inv = np.linalg.inv(self._lidar_pose_mats[self.start_index])

        angle_mode = self.lidar_angle_mode.strip().lower()
        if angle_mode == "fixed" and self.lidar_vertical_angles_deg is None:
            self.lidar_vertical_angles_deg = list(PANDAR64_VERT_DEG)
        if angle_mode == "fixed" and self.lidar_row_azimuth_offsets_deg is None:
            self.lidar_row_azimuth_offsets_deg = list(PANDAR64_ROT_DEG)
        elif angle_mode == "fitted":
            self.lidar_vertical_angles_deg = self._estimate_vertical_angles(
                frames=min(20, self.segment_length),
                min_range=max(self.near_plane, 2.0),
                far_plane=self.far_plane,
            )
        if self.lidar_row_azimuth_offsets_deg is None and self.lidar_vertical_angles_deg is not None:
            self.lidar_row_azimuth_offsets_deg = [0.0] * len(self.lidar_vertical_angles_deg)
        self._apply_vertical_angle_offset()

    def __len__(self) -> int:
        return self.segment_length

    def __getitem__(self, index: int) -> FrameSample:
        frame_idx = self.start_index + int(index)

        image = self._camera[frame_idx].convert("RGB")
        rgb = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).to(dtype=torch.float32) / 255.0

        points_world = self._lidar[frame_idx][["x", "y", "z"]].to_numpy(dtype=np.float32)
        points_lidar = self._world_to_sensor(points_world, self._lidar_pose_mats[frame_idx])
        points_lidar = self._filter_points(points_lidar)
        lidar_points = torch.from_numpy(points_lidar.astype(np.float32))

        lidar_depth = points_to_angle_table_depth(
            points_lidar=points_lidar,
            width=self.lidar_width,
            height=self.lidar_height,
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            vertical_fov_min_deg=self.lidar_vertical_fov_min_deg,
            vertical_fov_max_deg=self.lidar_vertical_fov_max_deg,
            vertical_angles_deg=self.lidar_vertical_angles_deg,
            row_azimuth_offsets_deg=self.lidar_row_azimuth_offsets_deg,
        )

        lidar_to_world = torch.tensor(self._reference_inv @ self._lidar_pose_mats[frame_idx], dtype=torch.float32)
        camera_to_world = torch.tensor(self._reference_inv @ self._camera_pose_mats[frame_idx], dtype=torch.float32)
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
            frame_id=f"{self.sequence_id}_{frame_idx:03d}_{self.camera_name}",
        )

    def _estimate_vertical_angles(
        self,
        frames: int,
        min_range: float,
        far_plane: float,
    ) -> list[float]:
        elevations: list[np.ndarray] = []
        for local_idx in range(frames):
            frame_idx = self.start_index + local_idx
            points_world = self._lidar[frame_idx][["x", "y", "z"]].to_numpy(dtype=np.float32)
            points_lidar = self._world_to_sensor(points_world, self._lidar_pose_mats[frame_idx])
            if points_lidar.size == 0:
                continue
            r = np.linalg.norm(points_lidar, axis=1)
            valid = (r >= float(min_range)) & (r <= float(far_plane))
            if not np.any(valid):
                continue
            pts = points_lidar[valid]
            xy = np.linalg.norm(pts[:, :2], axis=1)
            elevation = np.arctan2(pts[:, 2], np.clip(xy, 1.0e-6, None))
            elevations.append(elevation)

        if not elevations:
            raise ValueError(f"Failed to estimate PandaSet vertical angles for sequence {self.sequence_id}")

        all_elev = np.concatenate(elevations, axis=0)
        quantiles = (np.arange(self.lidar_height, dtype=np.float32) + 0.5) / float(self.lidar_height)
        angles = np.degrees(np.quantile(all_elev, quantiles)).astype(np.float32)
        return sorted(angles.tolist(), reverse=True)

    def _apply_vertical_angle_offset(self) -> None:
        if self.lidar_vertical_angles_deg is None or not len(self.lidar_vertical_angles_deg):
            return
        offset = float(self.lidar_vertical_angle_offset_deg)
        if abs(offset) <= 1.0e-9:
            return
        self.lidar_vertical_angles_deg = [float(a) + offset for a in self.lidar_vertical_angles_deg]

    def _filter_points(self, points_lidar: np.ndarray) -> np.ndarray:
        if points_lidar.size == 0:
            return points_lidar.reshape(0, 3)
        r = np.linalg.norm(points_lidar, axis=1)
        valid = (r >= self.near_plane) & (r <= self.far_plane) & (r <= self.max_range)
        return points_lidar[valid]

    @staticmethod
    def _resolve_source_root(source_root: Path) -> Path:
        root = source_root
        if (root / "pandaset").is_dir() and not any((root / token).is_dir() for token in ("001", "002", "010")):
            root = root / "pandaset"
        if not root.exists():
            raise FileNotFoundError(f"PandaSet source path not found: {root}")
        return root

    @staticmethod
    def _camera_intrinsics_to_matrix(intrinsics: object) -> np.ndarray:
        fx = float(getattr(intrinsics, "fx"))
        fy = float(getattr(intrinsics, "fy"))
        cx = float(getattr(intrinsics, "cx"))
        cy = float(getattr(intrinsics, "cy"))
        return np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _pose_dict_to_matrix(pose: dict[str, dict[str, float]]) -> np.ndarray:
        heading = pose["heading"]
        position = pose["position"]
        quat_wxyz = np.array(
            [heading["w"], heading["x"], heading["y"], heading["z"]],
            dtype=np.float32,
        )
        rotation = PandaSetDataset._quat_wxyz_to_rotmat(quat_wxyz)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array(
            [position["x"], position["y"], position["z"]],
            dtype=np.float32,
        )
        return transform

    @staticmethod
    def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
        w, x, y, z = quat.astype(np.float32).tolist()
        n = np.sqrt(w * w + x * x + y * y + z * z)
        if n <= 1.0e-12:
            return np.eye(3, dtype=np.float32)
        w /= n
        x /= n
        y /= n
        z /= n
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _world_to_sensor(points_world: np.ndarray, sensor_to_world: np.ndarray) -> np.ndarray:
        if points_world.size == 0:
            return points_world.reshape(0, 3)
        world_to_sensor = np.linalg.inv(sensor_to_world)
        return (world_to_sensor[:3, :3] @ points_world.T + world_to_sensor[:3, [3]]).T.astype(np.float32)
