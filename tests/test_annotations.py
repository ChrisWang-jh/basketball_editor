import json

import pytest

from basketeditor.annotations import (
    load_annotations,
    save_annotations,
    validate_annotations,
)
from basketeditor.models import VideoInfo


def data(tmp_path):
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"video")
    info = VideoInfo(str(video), 30, 100, 640, 480, 100 / 30)
    annotations = [
        {"target": t, "frame_idx": i * 10, "x": 50, "y": 50, "label": 1}
        for i, t in enumerate(("player", "hoop", "ball"))
    ]
    return info, annotations


def test_different_first_frames_are_valid_and_persistent(tmp_path):
    info, annotations = data(tmp_path)
    validate_annotations(annotations, info)
    path = save_annotations(tmp_path, info, annotations)
    assert load_annotations(tmp_path, info) == annotations
    assert json.loads(path.read_text())["schema_version"] == 1
    save_annotations(tmp_path, info, [])
    assert load_annotations(tmp_path, info) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("x", float("nan")),
        ("frame_idx", 100),
        ("frame_idx", 0.5),
        ("target", "other"),
        ("box", [30, 30, 10, 10]),
        ("label", 3),
    ],
)
def test_invalid_annotation_rejected(tmp_path, field, value):
    info, annotations = data(tmp_path)
    annotations[0][field] = value
    with pytest.raises(ValueError):
        validate_annotations(annotations, info)


def test_modified_video_does_not_inherit_old_annotations(tmp_path):
    info, annotations = data(tmp_path)
    save_annotations(tmp_path, info, annotations)
    from pathlib import Path

    Path(info.path).write_bytes(b"changed video")
    with pytest.raises(ValueError, match="变化"):
        load_annotations(tmp_path, info)
