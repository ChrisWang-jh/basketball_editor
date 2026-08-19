"""篮球有效进攻片段提取器的命令行入口。

运行前需安装 numpy、opencv-python、gradio、torch、torchvision 和 FFmpeg，
并准备官方 SAM2 仓库及匹配的模型权重。

示例：
    python run.py video/ --sam2-repo third_party/sam2

具体实现位于 ``basketeditor`` 包：
    ui.py       - 视频标注界面
    sam2.py     - SAM2 预览和视频追踪
    rules.py    - 补帧、平滑和进攻识别规则
    pipeline.py - 完整流程编排
    video.py    - 视频读写与结果导出
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any

from basketeditor.models import REPO_DIR
from basketeditor.ui import create_interface
from basketeditor.video import discover_videos


def load_runtime_libraries() -> tuple[Any, Any, Any]:
    """加载可选运行依赖，并在缺失时提供明确提示。"""
    missing: list[str] = []
    loaded: dict[str, Any] = {}
    for name, package_name in (
        ("numpy", "numpy"),
        ("cv2", "opencv-python"),
        ("gradio", "gradio"),
    ):
        try:
            loaded[name] = importlib.import_module(name)
        except (ImportError, OSError):
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            f"Missing runtime dependencies: {', '.join(missing)}. "
            "Install them before starting the application."
        )
    return loaded["numpy"], loaded["cv2"], loaded["gradio"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the target annotation, SAM2 tracking, and attack clipping UI."
    )
    parser.add_argument(
        "video_dir",
        type=Path,
        help="Directory recursively containing basketball videos",
    )
    parser.add_argument(
        "--sam2-repo",
        type=Path,
        default="third_party/sam2",
        help="Path to the cloned SAM2 repository",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_DIR / "checkpoints" / "sam2.1_hiera_base_plus.pt",
        help="SAM2 checkpoint path",
    )
    parser.add_argument(
        "--model-config",
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
        help="SAM2 model config",
    )
    parser.add_argument("--device", default="cuda", help="auto, cuda, cpu, or mps")
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=16666, help="Server port")
    parser.add_argument(
        "--share", action="store_true", help="Create a temporary public Gradio link"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        np, cv2, gr = load_runtime_libraries()
        video_root = args.video_dir.expanduser().resolve()
        video_paths = discover_videos(video_root)
        demo = create_interface(
            video_paths=video_paths,
            video_root=video_root,
            checkpoint=args.checkpoint.expanduser(),
            model_config=args.model_config,
            device=args.device,
            sam2_repo=args.sam2_repo.expanduser() if args.sam2_repo else None,
            np=np,
            cv2=cv2,
            gr=gr,
        )
        demo.queue(default_concurrency_limit=1).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
