# inference_python —— Python 快速推理

基于 `ultralytics` 的实时推理脚本，用于快速验证模型与后处理正确性。

| 文件 | 作用 |
| --- | --- |
| [realtime.py](./realtime.py) | 摄像头 / 视频 / 图片推理，实时画框并统计 FPS |

## 运行

```powershell
python inference_python/realtime.py --source 0                     # 摄像头
python inference_python/realtime.py --source demo.mp4              # 视频
python inference_python/realtime.py --source img.jpg --engine onnx # 用 ONNX Runtime 推理
```

- 依赖：`ultralytics`、`opencv-python`
- `--engine onnx` 需先导出 `models/yolo11n_coco128.onnx`（见 `training/`）
