"""服务配置：路径解析与默认推理参数。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# 让 ultralytics 的配置/字体缓存落在工作区内（与训练脚本保持一致）
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

DEFAULT_PT = ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"
DEFAULT_ONNX = ROOT / "models" / "yolo11n_coco128.onnx"

IMGSZ = 416
CONF = 0.25


def allowed_origins() -> list[str]:
    """CORS 白名单：仅放行本地开发与桌面端来源。

    - Vite dev server（127.0.0.1 / localhost 的 5173）
    - Electron 以 file:// 加载页面时，浏览器发送的 Origin 为字面量 "null"

    可用环境变量 YOLO_ALLOWED_ORIGINS（逗号分隔）整体覆盖。
    """
    env = os.environ.get("YOLO_ALLOWED_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return ["http://127.0.0.1:5173", "http://localhost:5173", "null"]


ALLOWED_ORIGINS = allowed_origins()


def resolve_weights() -> str:
    """按优先级选择权重：ONNX > 训练产出的 .pt > ultralytics 默认权重。"""
    if DEFAULT_ONNX.exists():
        return str(DEFAULT_ONNX)
    if DEFAULT_PT.exists():
        return str(DEFAULT_PT)
    return "yolo11n.pt"
