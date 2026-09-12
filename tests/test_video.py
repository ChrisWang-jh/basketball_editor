import json
import shutil
import subprocess

import pytest

from basketeditor.models import AttackEvent, FrameState, TrackPoint
from basketeditor.stitch import stitch_clips
from basketeditor.video import (
    cut_clips,
    export_debug_video,
    read_frame_timestamps,
    read_video_info,
)

pytestmark = pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg required"
)


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def make_video(path, audio=False):
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=161x121:rate=15",
    ]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    command += ["-t", "1", "-c:v", "ffv1", "-pix_fmt", "bgr0", str(path)]
    subprocess.run(command, check=True, capture_output=True)


def test_exact_clip_frames_audio_and_odd_dimensions(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "odd.mkv"
    make_video(source, audio=True)
    info = read_video_info(source, cv2)
    read_frame_timestamps(info)
    event = AttackEvent(2, 3, 4, 2, 6)
    clips = cut_clips(info, [event], tmp_path)
    data = probe(clips[0])
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    assert int(video["nb_frames"]) == 5
    assert video["width"] == 162 and video["height"] == 122
    assert video["pix_fmt"] == "yuv420p"
    assert any(s["codec_type"] == "audio" for s in data["streams"])
    assert not list(tmp_path.glob("*.encoding.mp4"))


def test_stitch_audio_and_silent_clips_in_order(tmp_path):
    first, second = tmp_path / "first.mkv", tmp_path / "second.mkv"
    make_video(first, True)
    make_video(second, False)
    output = stitch_clips([first, second], tmp_path)
    data = probe(output)
    assert abs(float(data["format"]["duration"]) - 2) < 0.15
    assert {s["codec_type"] for s in data["streams"]} == {"video", "audio"}
    assert output.name.startswith("stitched_")


def test_variable_frame_timestamps_control_clip_duration(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "variable.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=30:duration=1",
            "-vf",
            r"setpts=if(lt(N\,10)\,N*2\,N+10)/(30*TB)",
            "-vsync",
            "vfr",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    info = read_video_info(source, cv2)
    # Container frame counts for VFR may be estimates; the pipeline reconciles them.
    info.frame_count = len(
        [
            f
            for f in json.loads(
                subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-show_frames",
                        "-of",
                        "json",
                        str(source),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
            )["frames"]
            if f.get("media_type") == "video"
        ]
    )
    read_frame_timestamps(info)
    assert info.frame_time(5) > 5 / 30 + 0.1
    assert info.frame_time(10) > info.frame_time(5)
    assert info.frame_time(info.frame_count - 1, end=True) == info.duration


@pytest.mark.parametrize("audio", [True, False])
def test_debug_video_keeps_audio_and_renders_only_current_observations(tmp_path, audio):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "source.mkv"
    make_video(source, audio=audio)
    info = read_video_info(source, cv2)
    read_frame_timestamps(info)
    masks_dir = tmp_path / "masks"
    (masks_dir / "player").mkdir(parents=True)
    mask = np.zeros((info.height, info.width), dtype=np.uint8)
    mask[80:110, 40:70] = 255
    observations = {target: [] for target in ("player", "hoop", "ball")}
    states = []
    for index in range(info.frame_count):
        # Stale files remain on disk deliberately: missing and estimated frames
        # must not get an instance mask just because a file exists.
        assert cv2.imwrite(str(masks_dir / "player" / f"{index:08d}.png"), mask)
        player = TrackPoint(
            index, (40, 80, 69, 109), (54.5, 94.5), True, 0.95, source="reacquired"
        )
        if index % 3 == 1:
            player = TrackPoint(index, source="lost")
        elif index % 3 == 2:
            player.interpolated = True
            player.source = "interpolated"
        for target, points in observations.items():
            points.append(player if target == "player" else TrackPoint(index))
        states.append(
            FrameState(
                index, **{target: values[-1] for target, values in observations.items()}
            )
        )
    output = tmp_path / "debug.mp4"
    export_debug_video(info, states, output, masks_dir, np, cv2, observations)
    data = probe(output)
    streams = {stream["codec_type"]: stream for stream in data["streams"]}
    assert ("audio" in streams) == audio
    assert streams["video"]["codec_name"] == "h264"
    assert int(streams["video"]["nb_frames"]) == info.frame_count
    assert streams["video"]["width"] == 162 and streams["video"]["height"] == 122
    before, after = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    try:
        for index in range(3):
            ok, original = before.read()
            assert ok
            ok, rendered = after.read()
            assert ok
            original_crop = original[90:100, 50:60].astype(float)
            rendered_crop = rendered[90:100, 50:60].astype(float)
            if index == 0:
                expected = original_crop * 0.58 + np.asarray([80, 211, 52]) * 0.42
                assert np.abs(rendered_crop - expected).mean() < 12
            else:
                assert np.abs(rendered_crop - original_crop).mean() < 12
    finally:
        before.release()
        after.release()
    assert not list(tmp_path.glob("*.encoding.mp4"))


def test_debug_replays_variable_rate_holds_without_speeding_up_audio(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    source = tmp_path / "variable_audio.mkv"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=gray:size=320x240:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2",
            "-vf",
            "setpts=2*PTS",
            "-vsync",
            "vfr",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    info = read_video_info(source, cv2)
    info.frame_count = 10
    read_frame_timestamps(info)
    states = [
        FrameState(index, *(TrackPoint(index) for _ in range(3))) for index in range(10)
    ]
    output = tmp_path / "debug_variable.mp4"
    export_debug_video(info, states, output, tmp_path / "missing_masks", np, cv2)
    streams = {stream["codec_type"]: stream for stream in probe(output)["streams"]}
    assert abs(float(streams["video"]["duration"]) - info.duration) < 0.11
    assert abs(float(streams["audio"]["duration"]) - info.duration) < 0.11
    assert int(streams["video"]["nb_frames"]) == round(info.duration * info.fps)
