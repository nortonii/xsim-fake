# 3DGUT + gsplat LiDAR/RGB Joint Training Framework

这个版本针对双位姿传感器设置：每一帧有一个 RGB 相机位姿和一个 LiDAR 位姿。RGB 走透视 `gsplat` 渲染，LiDAR 走单次球面 range-image 渲染，不再依赖多视图拼接。

## 传感器假设

- RGB: 普通 pinhole 相机，使用 `intrinsics + camera_c2w`
- LiDAR: 水平 360 度，俯仰有限视场，例如 `-15` 到 `15` 度
- LiDAR 监督: 一张 `1 x H x W` 的深度图，按球面角度展开后的 range image

## 数据格式

manifest 文件中每一行指向一个 JSON 元数据文件。示例：

```json
{
  "frame_id": "000001",
  "rgb_path": "data/rgb/000001.pt",
  "lidar_depth_path": "data/lidar/000001.npy",
  "camera_c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
  "lidar_c2w": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
  "intrinsics": [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
  "rgb_width": 640,
  "rgb_height": 480,
  "lidar_width": 1800,
  "lidar_height": 64
}
```

约定：

- `rgb_path` 可以是 `.pt` 或 `.npy`，内容为 `3 x H x W`
- `lidar_depth_path` 可以是 `.pt` 或 `.npy`，内容为 `H x W` 或 `1 x H x W`
- `camera_c2w` 和 `lidar_c2w` 为各自传感器到世界坐标的位姿
- `lidar_width` 对应水平 360 度展开宽度
- `lidar_height` 对应垂直通道数或投影高度

## 训练入口

```bash
python train.py --config configs/example.yaml
```

## 实现说明

- RGB 渲染器位于 `src/three_dgut_gsplat/model.py`，直接调用 `gsplat`
- LiDAR 渲染器同样位于 `src/three_dgut_gsplat/model.py`，将高斯中心一次性投影到球面 range-image，并做单次 splat
- 当前 LiDAR 渲染是纯 PyTorch 版本，优点是结构清晰、方便改投影模型；如果后续追求速度，再把这部分下沉到 CUDA/gsplat 内核
