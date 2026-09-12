import numpy as np
import pytest

from basketeditor.models import TrackingOptions, TrackPoint, VideoInfo
from basketeditor.sam2 import mask_to_track_point, select_component
from basketeditor.tracking import prepare_tracks, tracking_diagnostics


def point(i, x=100, y=100, anchored=False):
    return TrackPoint(
        i, (x - 10, y - 4, x + 10, y + 4), (x, y), True, 0.95, anchored=anchored
    )


def fixture(n=15):
    info = VideoInfo("test", 30, n, 400, 300, n / 30)
    tracks = {
        t: [point(i, anchored=i == 0) for i in range(n)]
        for t in ("player", "hoop", "ball")
    }
    return info, tracks


def test_short_gaps_only_and_provenance():
    info, tracks = fixture()
    tracks["ball"][2] = TrackPoint(2)
    for i in range(5, 12):
        tracks["ball"][i] = TrackPoint(i)
    result = prepare_tracks(tracks, info)
    assert result["ball"][2].visible and result["ball"][2].interpolated
    assert result["ball"][2].confidence < 0.5
    assert all(not p.visible for p in result["ball"][5:12])
    assert not tracks["ball"][2].visible
    assert tracking_diagnostics(result, info)["ball"]["missing_frames"] == 7


def test_unobserved_prefix_never_filled():
    info, tracks = fixture()
    tracks["hoop"][:5] = [TrackPoint(i) for i in range(5)]
    result = prepare_tracks(tracks, info)
    assert all(not p.visible for p in result["hoop"][:5])


def test_scene_cut_prevents_gap_interpolation():
    info, tracks = fixture()
    tracks["ball"][5:8] = [TrackPoint(i) for i in range(5, 8)]
    result = prepare_tracks(tracks, info, scene_cuts=[7])
    assert all(not p.visible for p in result["ball"][5:8])


def test_jitter_stabilized_pan_preserved():
    info, tracks = fixture()
    tracks["hoop"][7] = point(7, x=106)
    result = prepare_tracks(tracks, info)
    assert result["hoop"][7].center == (100, 100)
    tracks["hoop"] = [point(i, x=100 + 4 * i) for i in range(15)]
    result = prepare_tracks(tracks, info)
    assert result["hoop"][7].center == (128, 100)


def test_fixed_mode_preserves_anchor_and_stops_at_cut():
    info, tracks = fixture()
    tracks["hoop"][5:] = [TrackPoint(i) for i in range(5, 15)]
    result = prepare_tracks(
        tracks, info, TrackingOptions(hoop_mode="fixed"), scene_cuts=[10]
    )
    assert result["hoop"][8].visible
    assert result["hoop"][8].source == "fixed"
    assert not result["hoop"][11].visible


def test_wrong_mask_size_rejected_but_new_anchor_can_reacquire():
    info, tracks = fixture()
    tracks["ball"][4] = TrackPoint(4, (0, 0, 400, 300), (200, 150), True, 0.999)
    tracks["ball"][8] = point(8, x=350, anchored=True)
    result = prepare_tracks(tracks, info, TrackingOptions(ball_gap_seconds=0))
    assert not result["ball"][4].visible
    assert result["ball"][8].center == (350, 100)


def test_component_uses_clicked_object_instead_of_mask_union():
    cv2 = pytest.importorskip("cv2")
    mask = np.full((100, 100), -10, dtype=np.float32)
    mask[10:15, 10:15] = 6
    mask[40:95, 40:95] = 9
    selected = select_component(mask, np, cv2, [{"x": 12, "y": 12, "label": 1}])
    result = mask_to_track_point(0, selected, np)
    assert result.bbox == (10, 10, 14, 14)


def test_empty_and_invalid_masks():
    assert not mask_to_track_point(0, np.zeros((5, 5)), np).visible
    with pytest.raises(RuntimeError):
        mask_to_track_point(0, np.zeros((2, 3, 4)), np)


def test_track_length_mismatch_is_explicit():
    info, tracks = fixture()
    tracks["ball"].pop()
    with pytest.raises(ValueError, match="length"):
        prepare_tracks(tracks, info)
