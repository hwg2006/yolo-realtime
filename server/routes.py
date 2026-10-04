"""HTTP / WebSocket 路由：只做请求解析与响应封装，推理交给 Detector。"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect

from .detector import get_detector, peek_detector
from .schemas import DetectResponse

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    """健康检查：Electron 据此判断后端是否就绪。"""
    detector = peek_detector()
    return {"status": "ok", "weights": detector.weights if detector else None}


@router.post("/api/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """上传单张图片，返回检测结果。"""
    data = await file.read()
    bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return DetectResponse(width=0, height=0, infer_ms=0.0, detections=[])
    h, w = bgr.shape[:2]
    dets, ms = await asyncio.to_thread(get_detector().detect, bgr)
    return DetectResponse(width=w, height=h, infer_ms=ms, detections=dets)


@router.websocket("/ws/detect")
async def ws_detect(ws: WebSocket) -> None:
    """客户端推送二进制 JPEG 帧，服务端逐帧返回 JSON。"""
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_bytes()
            bgr = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                await ws.send_json({"error": "decode failed"})
                continue
            h, w = bgr.shape[:2]
            dets, ms = await asyncio.to_thread(get_detector().detect, bgr)
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
