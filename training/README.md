# training —— 训练与导出

在 COCO128 上微调 YOLO11n（纯 CPU 可跑），并把权重导出为 ONNX。

| 文件 | 作用 |
| --- | --- |
| [train.py](./train.py) | 训练 YOLO11n，产出 `best.pt` 与 mAP 指标 |
| [export_onnx.py](./export_onnx.py) | 把 `best.pt` 导出为 ONNX 并复制到 `models/` |

## 运行

```powershell
python training/train.py
python training/export_onnx.py --weights runs/train/coco128_yolo11n/weights/best.pt
```

- 产物：`runs/train/coco128_yolo11n/weights/best.pt`、`models/yolo11n_coco128.onnx`
- 依赖：`ultralytics`（含 `torch`，CPU 版）
- 脚本会在导入 `ultralytics` 前，通过根目录 `project_config.py` 把 `YOLO_CONFIG_DIR` 重定向到工作区内，避免写入用户目录。
