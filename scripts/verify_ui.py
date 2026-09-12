"""Browser workflow verification in temporary storage; no GPU inference.

Analysis is a UI fixture. Persistence, clip discovery and FFmpeg stitching use
real code. Tracking accuracy is covered separately by verify_tracking.py.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from basketeditor.network import configure_local_proxy_bypass  # noqa: E402

configure_local_proxy_bypass()
import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from playwright.sync_api import expect, sync_playwright  # noqa: E402

from basketeditor.ui import create_interface  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video", type=Path, help="Optional local video for screenshots"
    )
    args = parser.parse_args()
    artifact = ROOT / "outputs/validation"
    artifact.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="basketedit_ui_") as temp:
        clip_root = Path(temp)
        for folder, color in (("game_a", "red"), ("game_b", "blue")):
            directory = clip_root / folder
            directory.mkdir()
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    f"color=c={color}:s=160x90:r=15:d=0.6",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(directory / "attack_001.mp4"),
                ],
                check=True,
            )
        source_video = clip_root / "source_video.mp4"
        if args.video:
            shutil.copyfile(args.video.expanduser().resolve(), source_video)
        else:
            # A fresh clone can verify its UI without a checked-in binary fixture.
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=640x360:rate=30:duration=1",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-pix_fmt",
                    "yuv420p",
                    str(source_video),
                ],
                check=True,
            )
        capture = cv2.VideoCapture(str(source_video))
        step_caption = f"{1 / capture.get(cv2.CAP_PROP_FPS):.2f}s"
        capture.release()
        second_video = clip_root / "second_video.mp4"
        shutil.copyfile(source_video, second_video)
        analysis_calls = 0

        def analysis_fixture(info, annotations, *args, **kwargs):
            nonlocal analysis_calls
            assert {p["target"] for p in annotations if p["label"] == 1} == {
                "player",
                "ball",
                "hoop",
            }
            analysis_calls += 1
            if analysis_calls == 1:
                raise ValueError("用于浏览器验收的可恢复错误")
            output = clip_root / "generated"
            output.mkdir()
            for name in ("attack_001.mp4", "debug_tracking.mp4"):
                shutil.copyfile(clip_root / "game_a/attack_001.mp4", output / name)
            events = output / "events.json"
            events.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "release_sec": 0.2,
                                "outcome": "rim_contact",
                                "clip_start_sec": 0,
                                "clip_end_sec": 0.6,
                            }
                        ]
                    }
                )
            )
            return "浏览器验收生成结果；此处未运行模型。", [
                str(output / "debug_tracking.mp4"),
                str(output / "attack_001.mp4"),
                str(events),
            ]

        demo = create_interface(
            [source_video, second_video],
            clip_root,
            ROOT / "checkpoints/sam2.1_hiera_base_plus.pt",
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "cpu",
            ROOT / "third_party/sam2",
            np,
            cv2,
            gr,
            clip_root,
        )
        demo.queue().launch(
            server_name="127.0.0.1",
            server_port=16669,
            prevent_thread_lock=True,
            quiet=True,
            **demo.basketedit_launch_kwargs,
        )
        try:
            with (
                patch("basketeditor.ui.run_pipeline", analysis_fixture),
                sync_playwright() as p,
            ):
                browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
                page = browser.new_page(
                    viewport={"width": 1440, "height": 1100}, device_scale_factor=1
                )
                page.set_default_timeout(20000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route(
                    "**/*",
                    lambda route: (
                        route.continue_()
                        if route.request.url.startswith(
                            ("http://127.0.0.1:", "data:", "blob:")
                        )
                        else route.abort()
                    ),
                )

                def screenshot(name):
                    page.wait_for_function(
                        "[...document.querySelectorAll('.pending')].every(node => !node.checkVisibility())"
                    )
                    page.screenshot(path=str(artifact / name), full_page=True)

                def frame_ready():
                    page.locator("#be-frame img").first.wait_for(state="visible")
                    page.wait_for_function(
                        "document.querySelector('#be-frame img')?.complete"
                    )

                def disable_preview():
                    page.get_by_text("更多设置", exact=True).click()
                    page.get_by_label("显示即时分割预览", exact=True).uncheck()
                    page.get_by_text("更多设置", exact=True).click()

                def switch_mode(text):
                    editing = text == "裁剪视频"
                    page.get_by_role(
                        "button", name="← 上一页" if editing else "下一页 →", exact=True
                    ).click()
                    page.get_by_role("heading", name=text, exact=True).wait_for()
                    expect(page.locator("#be-page-count")).to_contain_text(
                        "01 / 02" if editing else "02 / 02"
                    )
                    boundary = "← 上一页" if editing else "下一页 →"
                    expect(
                        page.get_by_role("button", name=boundary, exact=True)
                    ).to_be_disabled()

                def queue_starts_with(text):
                    expect(page.locator("#be-clip-queue label").first).to_contain_text(
                        text
                    )

                page.goto("http://127.0.0.1:16669", wait_until="domcontentloaded")
                page.get_by_role("heading", name="裁剪视频", exact=True).wait_for()
                assert page.locator("#be-editor").is_visible()
                assert not page.locator("#be-stitcher").is_visible()
                assert "框选" not in page.inner_text("body")
                assert page.get_by_role(
                    "button", name="← 上一页", exact=True
                ).is_disabled()
                frame_ready()
                assert not page.locator("#be-stitcher").is_visible()
                assert not page.locator("#be-results").is_visible()
                assert not page.get_by_label(
                    "显示即时分割预览", exact=True
                ).is_visible()
                assert page.get_by_role(
                    "button", name="先标记三个目标", exact=True
                ).is_disabled()
                screenshot("ui-desktop.png")
                disable_preview()
                page.get_by_role("button", name="1 帧 →", exact=True).click()
                expect(page.locator("#be-frame-caption")).to_contain_text(step_caption)
                picture = page.locator("#be-frame img").first
                bounds = picture.bounding_box()
                picture.click(
                    position={"x": bounds["width"] * 0.25, "y": bounds["height"] * 0.25}
                )
                expect(page.locator("#be-stats .is-ready")).to_have_count(1)
                assert list((clip_root / "annotations").glob("*.json"))
                switch_mode("拼接集锦")
                expect(page.locator("#be-library-status")).to_contain_text("共 2 个")
                assert not page.locator("#be-editor").is_visible()
                assert not page.locator("#be-stitch-result").is_visible()
                assert page.get_by_role(
                    "button", name="生成集锦", exact=True
                ).is_disabled()

                for i in range(2):
                    page.locator("#be-clip-gallery button.thumbnail-item").nth(
                        i
                    ).click()
                    page.get_by_role("button", name="加入集锦", exact=True).click()
                    expect(page.locator("#be-selected-clips")).to_contain_text(
                        f"已选 {i + 1} 段"
                    )
                page.get_by_role("button", name="↑ 上移", exact=True).click()
                queue_starts_with("game_b")
                new_clip = clip_root / "game_a/attack_002.mp4"
                shutil.copyfile(clip_root / "game_a/attack_001.mp4", new_clip)
                page.get_by_role("button", name="刷新片段", exact=True).click()
                expect(page.locator("#be-library-status")).to_contain_text("共 3 个")
                queue_starts_with("game_b")
                new_clip.unlink()
                page.get_by_role("button", name="刷新片段", exact=True).click()
                expect(page.locator("#be-library-status")).to_contain_text("共 2 个")

                switch_mode("裁剪视频")
                frame_ready()
                expect(page.locator("#be-frame-caption")).to_contain_text(step_caption)
                expect(page.locator("#be-stats .is-ready")).to_have_count(1)
                for count, target in ((2, "篮球"), (3, "篮圈")):
                    page.locator("#be-targets").get_by_text(target, exact=True).click()
                    bounds = picture.bounding_box()
                    picture.click(
                        position={
                            "x": bounds["width"] * (0.2 + count * 0.1),
                            "y": bounds["height"] * 0.4,
                        }
                    )
                    expect(page.locator("#be-stats .is-ready")).to_have_count(count)
                page.get_by_role("button", name="开始裁剪", exact=True).click()
                expect(page.locator("#be-run-status")).to_contain_text("裁剪未完成")
                assert not page.locator("#be-results").is_visible()
                expect(
                    page.get_by_role("button", name="开始裁剪", exact=True)
                ).to_be_enabled()
                page.get_by_role("button", name="开始裁剪", exact=True).click()
                expect(page.locator("#be-run-status")).to_contain_text("裁剪完成")
                page.locator("#be-results").wait_for(state="visible")
                assert not page.get_by_text("下载片段与报告", exact=True).is_visible()
                page.get_by_role("button", name="下一步：拼接 →", exact=True).click()
                expect(page.locator("#be-library-status")).to_contain_text("共 3 个")
                queue_starts_with("game_b")

                page.get_by_role("button", name="生成集锦", exact=True).click()
                expect(page.locator("#be-stitch-status")).to_contain_text("拼接完成")
                page.locator("#be-stitch-result").wait_for(state="visible")
                output = next(clip_root.glob("stitched_*.mp4"))
                capture = cv2.VideoCapture(str(output))
                ok, first = capture.read()
                assert (
                    ok
                    and float(first[:, :, 0].mean())
                    > float(first[:, :, 2].mean()) + 100
                )
                capture.set(
                    cv2.CAP_PROP_POS_FRAMES, capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
                )
                ok, last = capture.read()
                assert (
                    ok
                    and float(last[:, :, 2].mean()) > float(last[:, :, 0].mean()) + 100
                )
                capture.release()
                with page.expect_download() as downloaded:
                    page.get_by_text("下载集锦", exact=True).click()
                assert Path(downloaded.value.path()).read_bytes() == output.read_bytes()
                screenshot("ui-clips-desktop.png")
                page.get_by_role("button", name="移除", exact=True).click()
                expect(page.locator("#be-selected-clips")).to_contain_text("已选 1 段")
                page.get_by_role("button", name="清空", exact=True).click()
                expect(page.locator("#be-selected-clips")).to_contain_text(
                    "还没有选择片段"
                )
                assert page.get_by_role(
                    "button", name="生成集锦", exact=True
                ).is_disabled()

                page.reload(wait_until="domcontentloaded")
                frame_ready()
                expect(page.locator("#be-stats .is-ready")).to_have_count(3)
                disable_preview()
                page.get_by_role("button", name="1 帧 →", exact=True).click()
                expect(page.locator("#be-frame-caption")).to_contain_text(step_caption)
                page.get_by_role("button", name="撤销", exact=True).click()
                expect(page.locator("#be-stats .is-ready")).to_have_count(2)
                page.locator("#be-source input").click()
                page.get_by_role("option", name="second_video.mp4", exact=True).click()
                expect(page.locator("#be-stats .is-ready")).to_have_count(0)
                page.locator("#be-source input").click()
                page.get_by_role("option", name="source_video.mp4", exact=True).click()
                expect(page.locator("#be-stats .is-ready")).to_have_count(2)

                page.set_viewport_size({"width": 390, "height": 844})
                frame_ready()
                screenshot("ui-mobile.png")
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth + 2"
                )
                switch_mode("拼接集锦")
                page.locator("#be-library").wait_for(state="visible")
                page.locator("#be-clip-gallery button.thumbnail-item").first.click()
                page.get_by_role("button", name="加入集锦", exact=True).click()
                expect(page.locator("#be-selected-clips")).to_contain_text("已选 1 段")
                screenshot("ui-clips-mobile.png")
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= window.innerWidth + 2"
                )

                for clip in clip_root.glob("**/attack_*.mp4"):
                    clip.unlink()
                page.get_by_role("button", name="刷新片段", exact=True).click()
                page.get_by_text("还没有精彩片段", exact=True).wait_for()
                assert not page.locator("#be-clip-preview").is_visible()
                assert page.get_by_role(
                    "button", name="生成集锦", exact=True
                ).is_disabled()
                page.get_by_role("button", name="← 返回裁剪", exact=True).click()
                page.get_by_role("heading", name="裁剪视频", exact=True).wait_for()
                assert not errors, errors
                print(
                    "PASS: previous/next pages, boundary buttons, preserved state, readiness, analysis UI fixture/error, real stitching and download, refresh, empty state, mobile; no GPU inference"
                )
                browser.close()
        finally:
            demo.close()


if __name__ == "__main__":
    main()
