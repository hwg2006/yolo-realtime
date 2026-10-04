"""项目级配置：仓库路径、运行环境与权重解析，供所有 Python 入口共用。

ultralytics 需要在被导入前就拿到 `YOLO_CONFIG_DIR`，因此本模块在被导入时即完成设置。
`server` 包直接 `from project_config import ...` 即可（服务从仓库根目录启动）；
`training/` 与 `inference_python/` 下的脚本以文件方式运行，其 `sys.path` 首项是脚本
所在目录，需先手动把仓库根目录加入 `sys.path` 再导入本模块。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（本文件所在目录）
ROOT = Path(__file__).resolve().parent

# 让 ultralytics 的配置/字体缓存落在工作区内，避免写入用户目录被沙箱拦截
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

MODELS_DIR = ROOT / "models"
# 训练产出的权重
DEFAULT_PT = ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"
# 导出并复制到 models/ 统一管理的 ONNX
DEFAULT_ONNX = MODELS_DIR / "yolo11n_coco128.onnx"


def weights_by_priority() -> str:
    """按优先级选择权重：导出 ONNX > 训练产出 .pt > ultralytics 默认权重。"""
    if DEFAULT_ONNX.exists():
        return str(DEFAULT_ONNX)
    if DEFAULT_PT.exists():
        return str(DEFAULT_PT)
    return "yolo11n.pt"
