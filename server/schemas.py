"""接口数据模型：前后端共享的响应契约。"""

from __future__ import annotations

from pydantic import BaseModel


class Detection(BaseModel):
    """单个检测框。"""

    cls: int
    label: str
    conf: float
    xyxy: list[float]


class DetectResponse(BaseModel):
    """一帧/一张图的检测结果。"""

    width: int
    height: int
    infer_ms: float
    detections: list[Detection]
