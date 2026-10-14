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
# 官方 YOLO26s 预训练权重（全 COCO 80 类）：精度显著高于 yolo11n（48.6 vs 39.5 mAP），
# CPU 单帧约 90ms，作为默认
OFFICIAL_PT = ROOT / "yolo26s.pt"
OFFICIAL_ONNX = MODELS_DIR / "yolo26s.onnx"
# 兜底：旧版官方 yolo11n（未下载 yolo26s 时使用）
FALLBACK_PT = ROOT / "yolo11n.pt"
FALLBACK_ONNX = MODELS_DIR / "yolo11n.onnx"
# 自训练（COCO128 微调）产出，识别能力较弱，仅显式指定时使用
DEFAULT_PT = ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"
DEFAULT_ONNX = MODELS_DIR / "yolo11n_coco128.onnx"


def weights_by_priority() -> str:
    """按优先级选择权重：官方 YOLO26s（ONNX 优先于 .pt）> 兜底 yolo11n > 自训练。

    未下载 yolo26s 时自动回退到已有的 yolo11n，保证服务总能启动；
    需要指定其它权重时可通过各入口的 `--weights models/yolo11s.onnx` 显式覆盖。
    """
    for p in (
        OFFICIAL_ONNX,
        OFFICIAL_PT,
        FALLBACK_ONNX,
        FALLBACK_PT,
        DEFAULT_ONNX,
        DEFAULT_PT,
    ):
        if p.exists():
            return str(p)
    return "yolo26s.pt"
