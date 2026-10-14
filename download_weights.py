"""一键下载官方 YOLO 预训练权重，并导出 ONNX 到 models/。

仓库把 `*.pt` / `*.onnx` 排除在版本控制外（体积大），因此克隆下来后 `models/` 是空的、
无法直接推理——在仓库根目录执行本脚本即可补齐：

    python download_weights.py                        # 默认 yolo26s：下载 .pt + 导出 models/yolo26s.onnx @640
    python download_weights.py --models yolo26m.pt yolo11n.pt
    python download_weights.py --no-onnx              # 只下 .pt，不导出 ONNX
    python download_weights.py --force                # 已存在也重新下载 / 重新导出

权重来源为 ultralytics 官方 GitHub Release，与 `YOLO("yolo26s.pt")` 的自动下载同源。
导出参数与 `training/export_onnx.py` 保持一致（opset 12 / simplify / dynamic=False）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# 以文件方式运行时 sys.path 首项即本文件所在目录（仓库根），可直接导入共享配置
sys.path.insert(0, str(Path(__file__).resolve().parent))
from project_config import MODELS_DIR, ROOT  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils.downloads import attempt_download_asset  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="下载官方 YOLO 预训练权重并导出 ONNX")
    p.add_argument(
        "--models",
        nargs="+",
        default=["yolo26s.pt"],
        help="要下载的官方权重文件名，可多个（默认 yolo26s.pt，可选 yolo26n.pt / yolo11n.pt 等）",
    )
    p.add_argument("--imgsz", type=int, default=640, help="导出 ONNX 的输入尺寸（默认 640）")
    p.add_argument("--opset", type=int, default=12, help="ONNX opset（默认 12）")
    p.add_argument("--no-onnx", action="store_true", help="只下载 .pt，不导出 ONNX")
    p.add_argument("--force", action="store_true", help="已存在也强制重新下载 / 重新导出")
    return p.parse_args()


def download_pt(name: str, force: bool) -> Path:
    """下载官方 .pt 到仓库根目录（已存在则跳过），返回其绝对路径。"""
    dst = Path(name)
    if not dst.is_absolute():
        dst = ROOT / dst
    if force and dst.exists():
        dst.unlink()
    if dst.exists():
        print(f"[跳过] 已存在: {dst}")
        return dst

    print(f"[下载] {name} ...")
    attempt_download_asset(str(dst))
    if not dst.exists():
        raise SystemExit(
            f"下载失败: {name}（请检查网络，或换成 --models 指定其它权重）"
        )
    print(f"[完成] {dst}")
    return dst


def export_onnx(pt: Path, imgsz: int, opset: int, force: bool) -> Path:
    """把 .pt 导出为 ONNX 并放到 models/ 下统一管理。"""
    out = MODELS_DIR / (pt.stem + ".onnx")
    if out.exists() and not force:
        print(f"[跳过] 已存在: {out}（需重导加 --force）")
        return out

    print(f"[导出] {pt.name} -> {out} (imgsz={imgsz}, opset={opset})")
    exported = Path(
        YOLO(str(pt)).export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            simplify=True,
            dynamic=False,
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    if exported.resolve() != out.resolve():
        shutil.move(str(exported), str(out))
    print(f"[完成] {out}")
    return out


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        pt = download_pt(name, args.force)
        if not args.no_onnx:
            export_onnx(pt, args.imgsz, args.opset, args.force)

    print("\n权重已就绪，可启动本地摄像头实时推理：")
    print("  python inference_python/realtime.py --source 0")


if __name__ == "__main__":
    main()
