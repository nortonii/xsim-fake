from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from three_dgut_gsplat_min.data_pandaset import PandaSetDataset
from three_dgut_gsplat_min.lidar_projection import assign_ring_and_beam_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PandaSet ground-truth lidar depth.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pandaset" / "pandaset",
        help="PandaSet root directory.",
    )
    parser.add_argument("--sequence-id", type=str, default="001", help="PandaSet sequence ID.")
    parser.add_argument("--camera-name", type=str, default="front_camera", help="Camera name.")
    parser.add_argument("--lidar-sensor-id", type=int, default=0, help="PandaSet LiDAR sensor ID.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index inside the sequence.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "pandaset" / "exports",
        help="Directory to write the exported files.",
    )
    parser.add_argument(
        "--lidar-height",
        type=int,
        default=64,
        help="Vertical resolution of the exported depth image.",
    )
    parser.add_argument(
        "--lidar-width",
        type=int,
        default=2048,
        help="Horizontal resolution of the exported depth image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = PandaSetDataset(
        source_path=str(args.dataset_root),
        sequence_id=str(args.sequence_id),
        camera_name=str(args.camera_name),
        lidar_sensor_id=int(args.lidar_sensor_id),
        segment_length=max(int(args.frame_index) + 1, 1),
        lidar_width=int(args.lidar_width),
        lidar_height=int(args.lidar_height),
        lidar_angle_mode="fixed",
    )
    sample = dataset[int(args.frame_index)]

    export_dir = args.output_dir / f"pandaset_seq{args.sequence_id}_{args.camera_name}_frame{int(args.frame_index):03d}"
    export_dir.mkdir(parents=True, exist_ok=True)

    depth_path = export_dir / "gt_lidar_depth.npy"
    torch.save(sample.lidar_depth.detach().cpu(), export_dir / "gt_lidar_depth.pt")
    np.save(depth_path, sample.lidar_depth.detach().cpu().numpy())

    ring_map = np.full((int(sample.lidar_height), int(sample.lidar_width)), -1, dtype=np.int16)
    ring_vis = np.zeros((int(sample.lidar_height), int(sample.lidar_width), 3), dtype=np.uint8)
    if sample.lidar_points is not None and sample.lidar_points.numel() > 0:
        points = sample.lidar_points.detach().cpu().numpy().astype(np.float32)
        ring_id, beam_id, ranges, _ = assign_ring_and_beam_ids(
            points_lidar=points,
            width=int(sample.lidar_width),
            height=int(sample.lidar_height),
            near_plane=float(dataset.near_plane),
            far_plane=float(dataset.far_plane),
            vertical_fov_min_deg=float(dataset.lidar_vertical_fov_min_deg),
            vertical_fov_max_deg=float(dataset.lidar_vertical_fov_max_deg),
            vertical_angles_deg=list(dataset.lidar_vertical_angles_deg) if dataset.lidar_vertical_angles_deg is not None else None,
            row_azimuth_offsets_deg=list(dataset.lidar_row_azimuth_offsets_deg) if dataset.lidar_row_azimuth_offsets_deg is not None else None,
        )
        if ranges.size > 0:
            flat_idx = ring_id * int(sample.lidar_width) + beam_id
            order = np.argsort(ranges)
            flat_idx = flat_idx[order]
            ring_sorted = ring_id[order]
            ranges_sorted = ranges[order]
            seen = np.zeros(int(sample.lidar_height) * int(sample.lidar_width), dtype=bool)
            ring_flat = ring_map.reshape(-1)
            for idx, rid, _r in zip(flat_idx, ring_sorted, ranges_sorted, strict=False):
                if not seen[idx]:
                    ring_flat[idx] = int(rid)
                    seen[idx] = True

    valid_mask = ring_map >= 0
    if np.any(valid_mask):
        palette = np.array(
            [
                [0, 0, 0],
                [230, 25, 75],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
            ],
            dtype=np.uint8,
        )
        idx = np.mod(ring_map[valid_mask], len(palette))
        ring_vis[valid_mask] = palette[idx]
    Image.fromarray(ring_vis).save(export_dir / "gt_lidar_ring_ids.png")
    np.save(export_dir / "gt_lidar_ring_ids.npy", ring_map)

    meta = {
        "sequence_id": str(args.sequence_id),
        "camera_name": str(args.camera_name),
        "lidar_sensor_id": int(args.lidar_sensor_id),
        "frame_index": int(args.frame_index),
        "frame_id": sample.frame_id,
        "frame_timestamp": sample.frame_timestamp,
        "lidar_height": int(sample.lidar_height),
        "lidar_width": int(sample.lidar_width),
        "num_lidar_points": int(sample.lidar_points.shape[0]) if sample.lidar_points is not None else 0,
        "num_point_ring_ids": int(sample.lidar_point_ring_ids.shape[0]) if sample.lidar_point_ring_ids is not None else 0,
        "num_point_timestamps": int(sample.lidar_point_timestamps.shape[0]) if sample.lidar_point_timestamps is not None else 0,
    }
    (export_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote depth to {depth_path}")
    print(f"wrote ring map to {export_dir / 'gt_lidar_ring_ids.npy'}")
    print(f"wrote metadata to {export_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
