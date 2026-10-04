"""服务配置：默认推理参数与 CORS 白名单。

路径与权重解析统一由仓库根目录的 `project_config.py` 提供。
"""

from __future__ import annotations

import os

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
