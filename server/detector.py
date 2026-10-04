"""检测器：封装 ultralytics 模型，提供全局单例的加载与获取。"""

from __future__ import annotations

import time

import numpy as np

from .config import CONF, IMGSZ, resolve_weights
from .schemas import Detection


class Detector:
    """封装 ultralytics 模型，线程安全地执行推理。

    加载 .onnx 时，ultralytics 底层即使用 ONNX Runtime 作为推理后端。
    """

    def __init__(self, weights: str, imgsz: int = IMGSZ, conf: float = CONF) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names
        self.weights = weights

    def detect(self, bgr: np.ndarray) -> tuple[list[Detection], float]:
        """对一张 BGR 图像推理，返回检测框列表与耗时(ms)。"""
        t0 = time.perf_counter()
        result = self.model.predict(
            source=bgr, imgsz=self.imgsz, conf=self.conf, verbose=False
        )[0]
        infer_ms = (time.perf_counter() - t0) * 1000.0

        dets: list[Detection] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append(
                    Detection(
                        cls=int(k),
                        label=str(self.names.get(int(k), k)),
                        conf=float(c),
                        xyxy=[float(x1), float(y1), float(x2), float(y2)],
                    )
                )
        return dets, infer_ms


_service: Detector | None = None


def load_detector() -> Detector:
    """加载（或复用）全局检测器实例，供服务启动时调用。"""
    global _service
    if _service is None:
        weights = resolve_weights()
        print(f"[server] loading model: {weights}")
        _service = Detector(weights)
        print(f"[server] model ready, {len(_service.names)} classes")
    return _service


def get_detector() -> Detector:
    """获取已加载的检测器；未就绪时抛错，避免调用空实例。"""
    if _service is None:
        raise RuntimeError("detector 尚未加载")
    return _service


def peek_detector() -> Detector | None:
    """返回当前的检测器实例（可能为 None）。"""
    return _service
