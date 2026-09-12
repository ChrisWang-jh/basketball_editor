"""Rendered pixels must reflect accepted current-frame masks, not stale boxes."""

from dataclasses import replace

import pytest

from basketeditor.models import COLORS, FrameState, TrackPoint
from basketeditor.video import render_debug_frame

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")


def fixture_frame(target="player", source="sam2"):
    image = np.full((360, 640, 3), 120, dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[180:280, 80:110] = 255
    mask[250:280, 110:160] = 255
    points = {name: TrackPoint(0) for name in COLORS}
    points[target] = TrackPoint(
        0, (80, 180, 159, 279), (100, 230), True, 0.96, source=source
    )
    return image, mask, FrameState(0, **points)


@pytest.mark.parametrize("target", ["player", "hoop", "ball"])
def test_observed_instances_have_color_fill_contour_and_no_bbox(target):
    image, mask, frame = fixture_frame(target)
    rendered = render_debug_frame(image, frame, {target: mask}, np, cv2)
    expected = (image[220, 95] * 0.58 + np.asarray(COLORS[target][::-1]) * 0.42).astype(
        np.uint8
    )
    assert np.array_equal(rendered[220, 95], expected)
    # The top-right box corner lies outside this L-shaped instance. A rectangle
    # would color it, whereas a true segmentation leaves it alone.
    assert np.array_equal(rendered[180, 159], image[180, 159])
    assert not np.array_equal(rendered[220, 80], image[220, 80])
    assert np.array_equal(image[220, 95], [120, 120, 120])


@pytest.mark.parametrize(
    "source,visible,interpolated",
    [
        ("lost", False, False),
        ("pending", True, False),
        ("identity_rejected", True, False),
        ("future_unknown_source", True, False),
        ("interpolated", True, True),
        ("fixed", True, True),
    ],
)
def test_stale_masks_do_not_paint_lost_unverified_or_estimated_targets(
    source, visible, interpolated
):
    image, mask, frame = fixture_frame()
    original = frame.player
    frame.player = replace(
        original, source=source, visible=visible, interpolated=interpolated
    )
    rendered = render_debug_frame(
        image, frame, {"player": mask}, np, cv2, {"player": original}
    )
    assert np.array_equal(rendered[220, 95], image[220, 95])


@pytest.mark.parametrize("source", ["stabilized", "fixed"])
def test_smoothing_preserves_true_mask_at_the_observed_position(source):
    image, mask, frame = fixture_frame("hoop")
    original = frame.hoop
    frame.hoop = replace(original, bbox=(200, 180, 279, 279), source=source)
    rendered = render_debug_frame(
        image, frame, {"hoop": mask}, np, cv2, {"hoop": original}
    )
    assert not np.array_equal(rendered[220, 95], image[220, 95])
    assert np.array_equal(rendered[220, 215], image[220, 215])


def test_unobserved_fixed_target_does_not_reuse_previous_mask():
    image, mask, frame = fixture_frame("hoop", source="fixed")
    rendered = render_debug_frame(
        image, frame, {"hoop": mask}, np, cv2, {"hoop": TrackPoint(0, source="lost")}
    )
    assert np.array_equal(rendered[220, 95], image[220, 95])


@pytest.mark.parametrize("kind", ["wrong_size", "nan", "wrong_position"])
def test_corrupt_or_misaligned_mask_is_reported_and_hidden(kind):
    image, mask, frame = fixture_frame()
    if kind == "wrong_size":
        mask = mask[:-1]
    elif kind == "nan":
        mask = mask.astype(np.float32)
        mask[0, 0] = np.nan
    else:
        mask = np.roll(mask, 200, axis=1)
    with pytest.warns(RuntimeWarning, match="Ignoring"):
        rendered = render_debug_frame(image, frame, {"player": mask}, np, cv2)
    assert np.array_equal(rendered[180:, :], image[180:, :])


def test_reacquired_target_is_rendered_from_its_new_mask():
    image, mask, frame = fixture_frame(source="reacquired")
    rendered = render_debug_frame(image, frame, {"player": mask}, np, cv2)
    assert not np.array_equal(rendered[220, 95], image[220, 95])


def test_other_frame_observation_cannot_authorize_a_mask():
    image, mask, frame = fixture_frame()
    rendered = render_debug_frame(
        image,
        frame,
        {"player": mask},
        np,
        cv2,
        {"player": replace(frame.player, frame_idx=1)},
    )
    assert np.array_equal(rendered[220, 95], image[220, 95])
