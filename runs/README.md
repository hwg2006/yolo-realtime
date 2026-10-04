# runs

训练与推理的产物目录（权重、指标曲线、演示帧等），由脚本运行时生成；体积较大的部分不入库，见根 `.gitignore`。

- `runs/train/<name>/`：训练输出，含 `results.csv`、指标曲线与 `weights/best.pt`（`weights/` 不入库）。
- `runs/detect/`：Rust / Python 推理的标注图片输出（不入库）。

在仓库根目录重新生成：

```powershell
python training/train.py    # 产出 runs/train/coco128_yolo11n/
```
