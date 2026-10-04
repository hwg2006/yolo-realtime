"""HTTP / WebSocket 路由：只做请求解析与响应封装，推理交给 Detector。"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, WebSocket, WebSocketDisconnect

from .detector import get_detector, peek_detector
from .schemas import DetectResponse

router = APIRouter()


def decode_frame(payload: bytes):
    """把上传 / 推送的图像字节解码为 BGR 图，失败返回 None。"""
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


@router.get("/api/health")
def health() -> dict:
    """健康检查：Electron 据此判断后端是否就绪。"""
    detector = peek_detector()
    return {"status": "ok", "weights": detector.weights if detector else None}


@router.post("/api/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    """上传单张图片，返回检测结果。"""
    data = await file.read()
    bgr = decode_frame(data)
    if bgr is None:
        return DetectResponse(width=0, height=0, infer_ms=0.0, detections=[])
    dets, ms = await asyncio.to_thread(get_detector().detect, bgr)
    h, w = bgr.shape[:2]
    return DetectResponse(width=w, height=h, infer_ms=round(ms, 2), detections=dets)


@router.websocket("/ws/detect")
async def ws_detect(ws: WebSocket) -> None:
    """客户端推送二进制 JPEG 帧，服务端逐帧返回 JSON（与 HTTP 共用 DetectResponse）。"""
    await ws.accept()
    try:
        while True:
            payload = await ws.receive_bytes()
            bgr = decode_frame(payload)
            if bgr is None:
                resp = DetectResponse(width=0, height=0, infer_ms=0.0, detections=[])
            else:
                dets, ms = await asyncio.to_thread(get_detector().detect, bgr)
                h, w = bgr.shape[:2]
                resp = DetectResponse(width=w, height=h, infer_ms=round(ms, 2), detections=dets)
            await ws.send_json(resp.model_dump())
    except WebSocketDisconnect:
        pass

