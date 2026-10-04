"""应用组装：创建 FastAPI 实例、挂载中间件与路由、注册模型加载钩子。

推理逻辑见 detector.py，路由见 routes.py，数据模型见 schemas.py，
本文件只负责把各模块组装成一个可运行的服务。

启动:
    python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS
from .detector import load_detector
from .routes import router


def create_app() -> FastAPI:
    """组装并返回 FastAPI 应用。"""
    app = FastAPI(title="YOLO Realtime Inference Service")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.on_event("startup")
    def _load_model() -> None:
        load_detector()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
