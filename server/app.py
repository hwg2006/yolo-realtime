"""FastAPI 推理服务：为 Web / 桌面端提供实时检测能力。

- POST /api/detect : 上传单张图片，返回 JSON 检测结果
- WS   /ws/detect   : 客户端持续推送 JPEG 帧(二进制)，服务端返回 JSON 检测结果
- GET  /api/health  : 健康检查

启动:
    python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
import io
import os
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
# 让 ultralytics 的配置/字体缓存落在工作区内（与训练脚本保持一致）
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
DEFAULT_PT = ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"
DEFAULT_ONNX = ROOT / "models" / "yolo11n_coco128.onnx"

State = Literal["ok", "error"]


class Detection(BaseModel):
    cls: int
    label: str
    conf: float
    xyxy: list[float]


class DetectResponse(BaseModel):
    width: int
    height: int
    infer_ms: float
    detections: list[Detection]


def _resolve_weights() -> str:
    if DEFAULT_ONNX.exists():
        return str(DEFAULT_ONNX)
    if DEFAULT_PT.exists():
        return str(DEFAULT_PT)
    return "yolo11n.pt"


class Detector:
    """封装 ultralytics 模型，线程安全地执行推理。

    加载 .onnx 时，ultralytics 底层即使用 ONNX Runtime 作为推理后端。
    """

    def __init__(self, weights: str, imgsz: int = 416, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights, task="detect")
        self.imgsz = imgsz
        self.conf = conf
        self.names = self.model.names
        self.weights = weights

    def detect(self, bgr: np.ndarray) -> tuple[list[Detection], float]:
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


detector: Detector | None = None

app = FastAPI(title="YOLO Realtime Inference Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_model() -> None:
    global detector
    weights = _resolve_weights()
    print(f"[server] loading model: {weights}")
    detector = Detector(weights)
    print(f"[server] model ready, {len(detector.names)} classes")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "weights": detector.weights if detector else None}


@app.post("/api/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return DetectResponse(width=0, height=0, infer_ms=0.0, detections=[])
    h, w = bgr.shape[:2]
    dets, ms = await asyncio.to_thread(detector.detect, bgr)
    return DetectResponse(width=w, height=h, infer_ms=ms, detections=dets)


@app.websocket("/ws/detect")
async def ws_detect(ws: WebSocket) -> None:
    """客户端推送二进制 JPEG 帧，服务端返回 JSON。"""
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_bytes()
            bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                await ws.send_json({"error": "decode failed"})
                continue
            h, w = bgr.shape[:2]
            dets, ms = await asyncio.to_thread(detector.detect, bgr)
            await ws.send_json(
                {
                    "width": w,
                    "height": h,
                    "infer_ms": round(ms, 2),
                    "detections": [d.model_dump() for d in dets],
                }
            )
    except WebSocketDisconnect:
        pass


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
