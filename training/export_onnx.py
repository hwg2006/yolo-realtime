"""把训练好的 .pt 权重导出为 ONNX，供 Rust / Python(onnxruntime) / 后端服务使用。

用法:
    python training/export_onnx.py --weights runs/train/coco128_yolo11n/weights/best.pt
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
MODELS_DIR = ROOT / "models"

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"),
    )
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--opset", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"ONNX 已导出: {onnx_path}")

    # 复制一份到 models/ 统一管理
    import shutil

    dst = MODELS_DIR / "yolo11n_coco128.onnx"
    shutil.copy(onnx_path, dst)
    print(f"已复制到: {dst}")

    # 关键信息提示后续使用
    print(f"输入尺寸: {args.imgsz}x{args.imgsz}, 输出为 [1, 4+nc, 3549] 形式（imgsz=416, nc=80 → 84 通道）")


if __name__ == "__main__":
    main()
