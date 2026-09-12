from types import SimpleNamespace

import numpy as np
import pytest

from basketeditor import sam2
from basketeditor.models import ReIDOptions, VideoInfo


def test_targets_and_directions_use_independent_memory(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    frames = tmp_path / "frames"
    frames.mkdir()
    (frames / "00000000.jpg").write_bytes(b"not decoded by the fake predictor")
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    info = VideoInfo("test", 30, 10, 64, 64, 1 / 3)
    annotations = [
        {"target": name, "frame_idx": frame, "x": 20.0, "y": 20.0, "label": 1}
        for name, frame in [("player", 2), ("hoop", 7), ("ball", 4)]
    ]

    class Predictor:
        image_size = 32

        def __init__(self):
            self.runs = []
            self.current = None
            self.memory = False

        def init_state(self, **kwargs):
            return {}

        def reset_state(self, state):
            self.current = None
            self.memory = False

        def output(self, i):
            mask = np.full((1, 1, 64, 64), -8, dtype=np.float32)
            if self.current != 2 or i >= 5:
                mask[:, :, 16:25, 16:25] = 8
            return i, [self.current], mask

        def add_new_points_or_box(self, frame_idx, obj_id, **kwargs):
            assert self.current in (None, obj_id)
            self.current = obj_id
            return self.output(frame_idx)

        def propagate_in_video(
            self, state, start_frame_idx, max_frame_num_to_track, reverse
        ):
            assert not self.memory, "opposite direction reused dirty memory"
            self.memory = True
            self.runs.append((self.current, start_frame_idx, reverse))
            indices = (
                range(start_frame_idx, start_frame_idx - max_frame_num_to_track - 1, -1)
                if reverse
                else range(
                    start_frame_idx, start_frame_idx + max_frame_num_to_track + 1
                )
            )
            for i in indices:
                yield self.output(i)

    predictor = Predictor()
    original = sam2.importlib.import_module
    monkeypatch.setattr(sam2, "_load_torch", lambda repo: torch)
    monkeypatch.setattr(
        sam2.importlib,
        "import_module",
        lambda name: (
            SimpleNamespace(build_sam2_video_predictor=lambda *a, **k: predictor)
            if name == "sam2.build_sam"
            else original(name)
        ),
    )
    tracks = sam2.run_sam2_tracking(
        frames,
        info,
        annotations,
        checkpoint,
        "config",
        "cpu",
        None,
        np,
        reid_options=ReIDOptions(enabled=False),
    )
    assert predictor.runs == [
        (1, 2, False),
        (1, 2, True),
        (2, 7, False),
        (2, 7, True),
        (3, 4, False),
        (3, 4, True),
    ]
    assert all(not p.visible for p in tracks["hoop"][:5])
    assert all(p.visible for p in tracks["hoop"][5:])
    assert tracks["hoop"][7].anchored


def test_lazy_frames_match_official_normalization_and_bound_cache(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    for i in range(7):
        Image.new("RGB", (15, 10), (100 + i, 80, 50)).save(tmp_path / f"{i:08d}.jpg")
    frames = sam2.LazyVideoFrames(tmp_path, 7, 32, torch, np)
    for i in range(7):
        result = frames[i]
        assert result.shape == (3, 32, 32)
    assert len(frames.cache) == 4
    with Image.open(tmp_path / "00000006.jpg") as image:
        expected = torch.from_numpy(
            np.array(image.resize((32, 32)), dtype=np.float32) / 255
        ).permute(2, 0, 1)
    expected = (
        expected - torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    ) / torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    assert torch.allclose(result, expected)
    with pytest.raises(IndexError):
        frames[7]
