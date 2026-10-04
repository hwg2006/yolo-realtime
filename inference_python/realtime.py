"""Python 实时推理（快速实现版）。

支持摄像头 / 视频 / 图片 / 目录，实时绘制检测框并统计 FPS。

用法:
    python inference_python/realtime.py --source 0                       # 摄像头
    python inference_python/realtime.py --source demo.mp4                # 视频
    python inference_python/realtime.py --source image.jpg --save out/   # 图片
    python inference_python/realtime.py --source 0 --engine onnx         # 用 ONNX Runtime 推理
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# 以文件方式运行时 sys.path 首项是脚本目录，先补上仓库根目录再导入共享配置
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_config import DEFAULT_ONNX, DEFAULT_PT  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="0=摄像头 / 视频路径 / 图片路径 / 目录")
    p.add_argument("--engine", choices=["pt", "onnx"], default="pt")
    p.add_argument("--weights", default=None, help="自定义权重，默认取训练产出")
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--save", default=None, help="保存目录（不填则只显示）")
    p.add_argument("--no-show", action="store_true", help="不弹窗显示（无显示器环境）")
    return p.parse_args()


def resolve_weights(args: argparse.Namespace) -> str:
    if args.weights:
        return args.weights
    if args.engine == "onnx":
        return str(DEFAULT_ONNX if DEFAULT_ONNX.exists() else DEFAULT_PT)
    return str(DEFAULT_PT if DEFAULT_PT.exists() else "yolo11n.pt")


def normalize_source(src: str):
    return int(src) if src.isdigit() else src


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args)

    from ultralytics import YOLO

    # 直接用 ultralytics 加载；onnx 时其后端即 ONNX Runtime
    model = YOLO(weights, task="detect")

    source = normalize_source(args.source)
    show = not args.no_show
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    is_stream = source == 0 or str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
    writer = None
    fps, t_last, n = 0.0, time.time(), 0

    print(f"[realtime] engine={args.engine} weights={weights} source={source}")
    for result in model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        stream=is_stream,
        verbose=False,
    ):
        frame = result.plot()  # BGR 带框图像

        # FPS 估算（滑动平均）
        now = time.time()
        n += 1
        inst = 1.0 / max(now - t_last, 1e-6)
        fps = inst if fps == 0 else fps * 0.9 + inst * 0.1
        t_last = now
        cv2.putText(
            frame, f"FPS: {fps:5.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
        )
        if n % 30 == 0:
            print(f"frame={n} fps={fps:.1f} dets={len(result.boxes)}")

        if save_dir:
            cv2.imwrite(str(save_dir / f"frame_{n:05d}.jpg"), frame)

        if is_stream:
            if writer is None and isinstance(source, str):
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(save_dir / "output.mp4") if save_dir else "output.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h),
                )
            if writer:
                writer.write(frame)
            if show:
                cv2.imshow("YOLO realtime (python)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # 图片/目录：保存结果
            cv2.imwrite(str(save_dir / "result.jpg") if save_dir else "result.jpg", frame)
            print(f"[realtime] 检测到 {len(result.boxes)} 个目标，已保存结果")

    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[realtime] 完成，共处理 {n} 帧，平均 FPS≈{fps:.1f}")


if __name__ == "__main__":
    main()
