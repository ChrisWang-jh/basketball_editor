"""启动 attack 视频片段选择与拼接 UI。"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

from basketeditor.models import REPO_DIR
from basketeditor.network import configure_local_proxy_bypass
from basketeditor.stitch_ui import create_stitch_interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse, select, and stitch attack clips."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=REPO_DIR / "outputs",
        help="Directory recursively containing attack clips (default: outputs)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server address")
    parser.add_argument("--port", type=int, default=16667, help="Server port")
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_local_proxy_bypass(args.host)
    try:
        gr = importlib.import_module("gradio")
        output_root = args.output_dir.expanduser().resolve()
        demo = create_stitch_interface(output_root, gr)
        demo.queue(default_concurrency_limit=1).launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            **demo.basketedit_launch_kwargs,
        )
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
