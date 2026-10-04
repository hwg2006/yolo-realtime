"""YOLO 训练脚本（CPU 可跑）。

在 COCO128 小数据集上对 YOLO11n 进行微调，产出 best.pt 及训练指标。
用法:
    python training/train.py
    python training/train.py --model yolov8n.pt --epochs 30 --imgsz 416
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 让 ultralytics 的配置/字体缓存落在工作区内，避免写入 AppData 被沙箱拦截
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLO on COCO128 (CPU friendly)")
    p.add_argument("--model", default=str(ROOT / "yolo11n.pt"), help="预训练权重或模型配置")
    p.add_argument("--data", default=str(ROOT / "datasets" / "coco128.yaml"), help="数据集配置")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--name", default="coco128_yolo11n")
    return p.parse_args()


def main() -> None:
    # 切到仓库根目录，保证 coco128.yaml 中的相对数据集路径稳定解析
    os.chdir(ROOT)
    args = parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(ROOT / "runs" / "train"),
        name=args.name,
        exist_ok=True,
        cache=True,            # coco128 很小，缓存到内存加速
        pretrained=True,
        optimizer="auto",
        patience=20,
        seed=0,
        plots=True,
        val=True,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\n训练完成，权重: {best}")
    print(f"指标目录: {save_dir}")

    # 在测试集/验证集上跑一次 evaluate，打印 mAP
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print(f"mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
