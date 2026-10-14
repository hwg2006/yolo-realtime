"""服务配置：默认推理参数与 CORS 白名单。

路径与权重解析统一由仓库根目录的 `project_config.py` 提供。
"""

from __future__ import annotations

import os

# 推理输入尺寸：必须与导出 ONNX 时使用的固定尺寸一致（models/yolo11*.onnx 均为 640）
IMGSZ = 640
# 置信度阈值 / NMS IoU：沿用 ultralytics 官方默认（conf=0.25、iou=0.70），不做额外调参。
CONF = float(os.environ.get("YOLO_CONF", "0.25"))
IOU = float(os.environ.get("YOLO_IOU", "0.70"))


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
