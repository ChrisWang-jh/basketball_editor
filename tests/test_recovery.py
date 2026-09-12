"""Identity decisions must survive downstream smoothing and gap handling."""

from types import SimpleNamespace

import numpy as np
import pytest

from basketeditor.models import (
    TARGET_IDS,
    ReIDOptions,
    TrackingOptions,
    TrackPoint,
    VideoInfo,
)
from basketeditor.tracking import prepare_tracks


def observed(index, x=80.0, source="sam2", anchored=False):
    return TrackPoint(
        index,
        (x - 8, 50.0, x + 8, 80.0),
        (x, 65.0),
        True,
        0.95,
        source=source,
        anchored=anchored,
    )


def trajectories():
    info = VideoInfo("test", 30, 3, 640, 360, 0.1)
    tracks = {
        target: [observed(i, anchored=i == 0) for i in range(3)]
        for target in TARGET_IDS
    }
    return info, tracks


@pytest.mark.parametrize("target", TARGET_IDS)
@pytest.mark.parametrize("source", ["identity_rejected", "pending", "lost"])
def test_identity_veto_is_not_hidden_by_interpolation(target, source):
    info, tracks = trajectories()
    tracks[target][1] = TrackPoint(1, source=source)
    result = prepare_tracks(tracks, info)
    assert not result[target][1].visible
    assert result[target][1].source == source
    assert not result[target][1].interpolated


@pytest.mark.parametrize("source", ["identity_rejected", "pending", "lost"])
def test_fixed_hoop_cannot_overwrite_identity_veto(source):
    info, tracks = trajectories()
    tracks["hoop"][1] = TrackPoint(1, source=source)
    result = prepare_tracks(tracks, info, TrackingOptions(hoop_mode="fixed"))
    assert not result["hoop"][1].visible
    assert result["hoop"][1].source == source


def test_verified_recovery_can_return_away_from_last_position():
    info, tracks = trajectories()
    tracks["player"][1] = TrackPoint(1, source="lost")
    tracks["player"][2] = observed(2, x=550, source="reacquired")
    result = prepare_tracks(tracks, info)
    assert result["player"][2].visible
    assert result["player"][2].center == (550, 65)
    assert result["player"][2].source == "reacquired"


def test_verified_identity_does_not_excuse_invalid_geometry():
    info, tracks = trajectories()
    tracks["player"][2] = observed(2, x=1000, source="reacquired")
    result = prepare_tracks(tracks, info)
    assert not result["player"][2].visible


def test_unverified_short_sam2_hole_retains_legacy_interpolation():
    info, tracks = trajectories()
    tracks["ball"][1] = TrackPoint(1)
    result = prepare_tracks(tracks, info)
    assert result["ball"][1].visible
    assert result["ball"][1].interpolated


def run_identity_case(
    monkeypatch,
    tmp_path,
    objects,
    video_predictions,
    proposals,
    *,
    cuts=(),
    groups=None,
    partial_candidates=False,
    anchor_error=False,
    operations=None,
    audit_interval=100,
    target="player",
    features=None,
    live_audits=(),
    flow_support=None,
):
    """Exercise the real controller and matcher with observable model boundaries."""
    import cv2

    from basketeditor import reid, tracker

    frame_count = len(objects)
    features = features or {}
    for i in range(frame_count):
        cv2.imwrite(
            str(tmp_path / f"{i:08d}.jpg"), np.full((64, 80, 3), i * 16, np.uint8)
        )
    state = {"dirty": None}
    operations = [] if operations is None else operations
    searches = []

    def frame_index(rgb):
        return round(float(rgb[0, 0, 0]) / 16)

    def mask_at(x):
        mask = np.zeros((64, 80), bool)
        if x is not None:
            mask[30:46, x : x + 8] = True
        return mask

    class Extractor:
        def __init__(self, *args, **kwargs):
            pass

        def describe(self, rgb, mask, target=None):
            y, x = np.nonzero(mask)
            if len(x) == 0:
                return None
            index = frame_index(rgb)
            extra = features.get((index, int(x.min())), {})
            angle = extra.get("angle", objects[index].get(int(x.min()), np.pi / 2))
            angle = extra.get("angles_by_area", {}).get(len(x), angle)
            vector = np.zeros(8, np.float32)
            vector[0], vector[1] = np.cos(angle), np.sin(angle)
            color = np.zeros(64, np.float32)
            color[extra.get("color_index", 0)] = 1
            color = np.asarray(extra.get("color", color), np.float32)
            ball_vector = np.zeros(8, np.float32)
            ball_angle = extra.get("ball_angle", 0)
            ball_vector[0], ball_vector[1] = np.cos(ball_angle), np.sin(ball_angle)
            return reid.Appearance(
                vector,
                np.tile(vector, (3, 1)),
                color,
                np.tile(vector, (2, 1)),
                (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1),
                len(x),
                ball_vector,
                extra.get("shape", (0.8, 0.8, 0.75, 1.0)),
            )

        def search(self, rgb, memory, limit=6, previous_bbox=None, **kwargs):
            index = frame_index(rgb)
            if index in live_audits:
                assert state["dirty"] == index, (
                    "Audit interrupted a verified video stream"
                )
            else:
                assert state["dirty"] is None, (
                    "Recovery searched while rejected temporal memory was still live"
                )
            searches.append((index, previous_bbox, len(memory.anchors)))
            return [
                reid.Proposal((x + 4, 38), (x, 30, x + 8, 46), 0.99)
                for x in proposals.get(index, [])
            ]

        def release(self):
            operations.append(("release", None))

    class Segmenter:
        def __init__(self, predictor):
            pass

        def predict(self, rgb, index, *, points=None, proposal=None, small=False):
            if anchor_error:
                raise RuntimeError("anchor segmentation failed")
            operations.append(("segment", index))
            if proposal is not None:
                px, py = map(round, proposal.point)
            else:
                positive = next(p for p in points if p["label"] == 1)
                px, py = round(positive["x"]), round(positive["y"])
            if not (0 <= px < 80 and 0 <= py < 64):
                return []
            x = next(
                (
                    start
                    for start in objects[index]
                    if features.get((index, start), {}).get("mask", mask_at(start))[
                        py, px
                    ]
                ),
                None,
            )
            if x is None:
                # Motion proposals must not invent an object on an empty frame.
                return []
            extra = features.get((index, x), {})
            result = list(
                extra.get(
                    "candidates",
                    [(extra.get("mask", mask_at(x)), extra.get("quality", 0.99))],
                )
            )
            if proposal is not None and partial_candidates:
                part = mask_at(x)
                part[38:] = False
                result.append((part, 0.98))
            return result

    class Stream:
        def __init__(self, predictor, index, seed):
            self.predictor, self.index, self.start, self.seed = (
                predictor,
                index,
                index,
                seed,
            )
            self.closed = False

        def __next__(self):
            if self.closed or self.index >= frame_count:
                self.closed = True
                raise StopIteration
            index = self.index
            self.index += 1
            prediction = video_predictions.get(index)
            mask = (
                self.seed
                if index == self.start
                else (
                    prediction
                    if isinstance(prediction, np.ndarray)
                    else mask_at(prediction)
                )
            )
            state["dirty"] = index
            operations.append(("infer", index))
            return (
                index,
                [self.predictor.oid],
                np.where(mask[None, None], 8.0, -8.0).astype(np.float32),
            )

        def close(self):
            self.closed = True
            operations.append(("close", self.index))

    class Predictor:
        device = "cpu"

        def __init__(self):
            self.stream = None

        def reset_state(self, state):
            assert self.stream is None or self.stream.closed, (
                "reset_state called before closing the suspended generator"
            )
            state.clear()
            state["dirty"] = None
            operations.append(("reset", None))

        def add_new_mask(self, state, index, oid, mask):
            self.seed = mask
            self.oid = oid
            operations.append(("seed", index))

        def propagate_in_video(self, state, start_frame_idx, max_frame_num_to_track):
            self.stream = Stream(self, start_frame_idx, self.seed)
            return self.stream

    monkeypatch.setattr(reid, "DinoExtractor", Extractor)
    monkeypatch.setattr(tracker, "Segmenter", Segmenter)
    if flow_support is not None:

        def flow(previous_rgb, rgb, previous_mask, mask):
            if previous_rgb is None or rgb is None:
                return False
            index = frame_index(rgb)
            operations.append(("flow", (frame_index(previous_rgb), index)))
            return flow_support.get(index, False)

        monkeypatch.setattr(tracker, "_flow_support", flow)
    original_update = reid.IdentityMemory.update

    def record_update(memory, descriptor, frame_idx, verified=True):
        operations.append(("update", frame_idx))
        return original_update(memory, descriptor, frame_idx, verified)

    monkeypatch.setattr(reid.IdentityMemory, "update", record_update)
    groups = groups or {
        (target, 0): [{"target": target, "frame_idx": 0, "x": 12, "y": 38, "label": 1}]
    }
    diagnostics = {}
    tracks = tracker.track_identities(
        Predictor(),
        SimpleNamespace(
            reset_predictor=lambda: operations.append(("image_reset", None))
        ),
        state,
        tmp_path,
        VideoInfo("test", 30, frame_count, 80, 64, frame_count / 30),
        groups,
        ReIDOptions(audit_interval=audit_interval),
        None,
        cuts,
        None,
        diagnostics,
        None,
    )
    return tracks[target], operations, searches, diagnostics


def test_rejected_temporal_identity_is_reset_before_search_and_recovery(
    monkeypatch, tmp_path
):
    points, operations, searches, diagnostics = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {45: np.pi / 2}, {45: 0}, {45: 0}],
        video_predictions={1: 45, 3: 45},
        proposals={2: [45]},
    )
    assert not points[1].visible
    assert points[1].source == "identity_rejected"
    assert points[2].visible and points[2].source == "reacquired"
    assert points[2].center[0] > 40
    assert ("seed", 2) in operations
    assert all(anchor_count == 1 for _, _, anchor_count in searches)
    assert diagnostics["player"]["memory_resets"] >= 3


def test_cut_clears_motion_but_preserves_the_identity_anchor(monkeypatch, tmp_path):
    points, _, searches, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {8: 0}, {60: 0}],
        video_predictions={1: 8},
        proposals={2: [60]},
        cuts=[2],
    )
    assert points[2].visible and points[2].source == "reacquired"
    at_cut = [entry for entry in searches if entry[0] == 2]
    assert at_cut and all(
        previous is None and count == 1 for _, previous, count in at_cut
    )


def test_later_manual_anchor_can_identify_an_earlier_visible_frame(
    monkeypatch, tmp_path
):
    groups = {
        ("player", 2): [
            {"target": "player", "frame_idx": 2, "x": 12, "y": 38, "label": 1}
        ]
    }
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {8: 0}, {8: 0}],
        video_predictions={1: 8},
        proposals={0: [8]},
        groups=groups,
    )
    assert points[0].visible and points[0].source == "reacquired"
    assert points[2].anchored


def test_repeating_weak_appearance_is_not_new_identity_evidence(monkeypatch, tmp_path):
    # A stable look-alike remains a look-alike even after multiple frames.
    angle = 0.62
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {45: angle}, {45: angle}, {45: angle}],
        video_predictions={1: 45},
        proposals={1: [45], 2: [45], 3: [45]},
    )
    assert all(not point.visible for point in points[1:])


def test_large_jump_requires_independent_recovery_before_temporal_commit(
    monkeypatch, tmp_path
):
    points, _, searches, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {60: 0}],
        video_predictions={1: 60},
        proposals={1: [60]},
    )
    assert points[1].visible and points[1].source == "reacquired"
    assert any(index == 1 for index, _, _ in searches)


def test_negative_only_correction_rejects_both_temporal_and_retrieved_mask(
    monkeypatch, tmp_path
):
    groups = {
        ("player", 0): [
            {"target": "player", "frame_idx": 0, "x": 12, "y": 38, "label": 1}
        ],
        ("player", 1): [
            {"target": "player", "frame_idx": 1, "x": 64, "y": 38, "label": 0}
        ],
    }
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {60: 0}],
        video_predictions={1: 60},
        proposals={1: [60]},
        groups=groups,
    )
    assert not points[1].visible


def test_nested_masks_of_one_person_are_not_competing_identities(monkeypatch, tmp_path):
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {}, {45: 0}],
        video_predictions={},
        proposals={2: [45]},
        partial_candidates=True,
    )
    assert points[2].visible and points[2].source == "reacquired"


def test_two_indistinguishable_people_remain_ambiguous(monkeypatch, tmp_path):
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {}, {8: 0, 60: 0}],
        video_predictions={},
        proposals={2: [8, 60]},
        partial_candidates=True,
    )
    assert not points[2].visible
    assert points[2].reason == "ambiguous_lookalikes"


def test_unrelated_negative_does_not_turn_weak_impostor_into_identity(
    monkeypatch, tmp_path
):
    # The unrelated distractor at the manual frame does not add evidence about
    # a previously unseen look-alike appearing later at the old target position.
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0, 60: np.pi / 2}, {8: 0.62}, {8: 0.62}],
        video_predictions={1: 8, 2: 8},
        proposals={0: [60], 1: [8], 2: [8]},
    )
    assert all(not point.visible for point in points[1:])


def test_recovery_does_not_recycle_a_geometry_rejected_temporal_mask(
    monkeypatch, tmp_path
):
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {0: 0}],
        video_predictions={1: np.ones((64, 80), bool)},
        proposals={},
    )
    assert not points[1].visible


def test_anchor_failure_releases_both_feature_and_image_models(monkeypatch, tmp_path):
    operations = []
    with pytest.raises(RuntimeError, match="anchor segmentation failed"):
        run_identity_case(
            monkeypatch,
            tmp_path,
            objects=[{8: 0}],
            video_predictions={},
            proposals={},
            anchor_error=True,
            operations=operations,
        )
    assert ("release", None) in operations
    assert ("image_reset", None) in operations


@pytest.mark.parametrize(
    "positive,negative", [((1073, 304), (1067, 310)), ((4, 5), (18, 20))]
)
@pytest.mark.parametrize("small", [True, False])
def test_manual_small_target_roi_preserves_points_and_full_frame_mask(
    positive, negative, small
):
    from basketeditor.tracker import Segmenter

    y, x = np.indices((592, 1280))
    rgb = np.stack((x % 256, x // 256, y % 256), axis=2).astype(np.uint8)

    class RecordingPredictor:
        def set_image(self, image):
            self.image = image

        def predict(
            self, *, point_coords, point_labels, box, multimask_output, return_logits
        ):
            assert return_logits
            self.coords, self.labels = point_coords.copy(), point_labels.copy()
            logits = np.full(self.image.shape[:2], -8.0, np.float32)
            px, py = map(round, point_coords[0])
            logits[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2] = 8.0
            # Predicted IoU is deliberately low; valid mask logits determine
            # segmentation confidence and must survive coordinate restoration.
            return logits[None], np.asarray([0.01]), None

    model = RecordingPredictor()
    points = [
        {"x": px, "y": py, "label": label}
        for (px, py), label in ((positive, 1), (negative, 0))
    ]
    results = Segmenter(model).predict(rgb, 5, points=points, small=small)
    assert len(results) == 1
    mask, confidence = results[0]
    assert confidence > 0.99
    np.testing.assert_array_equal(model.labels, [1, 0])
    offset = np.asarray(positive) - model.coords[0]
    np.testing.assert_allclose(np.asarray(negative) - model.coords[1], offset)
    assert np.all(model.coords >= 0)
    assert np.all(model.coords < np.asarray(model.image.shape[1::-1]))
    if small:
        assert model.image.shape[0] < rgb.shape[0] / 2
        assert model.image.shape[1] < rgb.shape[1] / 2
    else:
        assert model.image.shape == rgb.shape
        np.testing.assert_array_equal(offset, [0, 0])
    x0, y0 = map(int, offset)
    np.testing.assert_array_equal(
        model.image, rgb[y0 : y0 + model.image.shape[0], x0 : x0 + model.image.shape[1]]
    )
    expected = np.zeros(rgb.shape[:2], bool)
    px, py = positive
    expected[py - 1 : py + 2, px - 1 : px + 2] = True
    np.testing.assert_array_equal(mask, expected)


def test_identity_audit_does_not_count_continuous_tracking_as_recovery(
    monkeypatch, tmp_path
):
    points, _, searches, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0} for _ in range(11)] + [{}, {8: 0}],
        video_predictions={index: 8 for index in range(1, 11)},
        proposals={10: [8], 12: [8]},
        audit_interval=10,
        live_audits=(10,),
    )
    assert any(index == 10 for index, _, _ in searches)
    assert points[10].visible and points[10].source == "sam2"
    assert not points[11].visible
    assert points[12].visible and points[12].source == "reacquired"
    assert [point.frame_idx for point in points if point.source == "reacquired"] == [12]


def test_audit_cannot_replace_a_continuous_target_with_a_distant_lookalike(
    monkeypatch, tmp_path
):
    # The real target changes pose (score about .89); a distant look-alike is
    # closer to the original anchor (about .996). An audit is not a new ID vote.
    points, _, _, diagnostics = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, *[{8: 0.5, 60: 0.1} for _ in range(4)]],
        video_predictions={index: 8 for index in range(1, 5)},
        proposals={index: [8, 60] for index in range(1, 5)},
        audit_interval=1,
        live_audits=range(1, 5),
        flow_support={1: True},
    )
    for point in points[1:]:
        assert point.visible and point.center[0] < 20
        assert point.source == "sam2"
        assert point.reason == "audit_competitor"
    assert diagnostics["player"]["templates"] == 0


@pytest.mark.parametrize("interruption", ["gap", "cut"])
def test_person_continuity_can_resolve_small_negative_margin_only_until_interrupted(
    monkeypatch, tmp_path, interruption
):
    objects = [
        {8: 0, 60: 0.9},
        {8: 0.15},
        {8: 0.30},
        {8: 0.43},
        {8: 0.43},
        {8: 0.43},
        {8: 0.43},
        {8: 0.43},
    ]
    if interruption == "gap":
        objects[5] = {}
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=objects,
        video_predictions={index: 8 for index in range(1, 5)},
        proposals={0: [60], 5: [8] if interruption == "cut" else [], 6: [8], 7: [8]},
        cuts=[5] if interruption == "cut" else [],
    )
    assert all(
        point.visible and point.reason == "motion_verified" for point in points[3:5]
    )
    assert not any(
        operation == "update" and index in (3, 4) for operation, index in operations
    )
    assert all(not point.visible for point in points[5:])


@pytest.mark.parametrize(
    "candidate_x,missing,expected", [(28, 0, True), (0, 0, False), (48, 2, True)]
)
def test_ball_continuity_requires_observed_velocity_not_only_nearby_color(
    monkeypatch, tmp_path, candidate_x, missing, expected
):
    index = 2 + missing
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=[{8: 0}, {18: 0}, *[{} for _ in range(missing)], {candidate_x: 0.9}],
        video_predictions={1: 18, index: candidate_x},
        proposals={index: [candidate_x]},
    )
    assert points[index].visible is expected
    if expected:
        assert points[index].reason == "motion_verified"
    assert ("update", index) not in operations


def test_ball_continuity_rejects_changed_shape_color_area_features_and_stale_history(
    monkeypatch, tmp_path
):
    enlarged = np.zeros((64, 80), bool)
    enlarged[20:52, 20:36] = True
    objects = [{8: 0}, {12: 0}] + [{8 + index * 4: 0.9} for index in range(2, 9)]
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=objects,
        video_predictions={1: 12, 2: 16, 3: enlarged},
        proposals={index: [8 + index * 4] for index in range(3, 9)},
        features={
            (3, 20): {"mask": enlarged},
            (4, 24): {"color_index": 1},
            (5, 28): {"shape": (0.2, 0.4, 0.1, 0.7)},
            (6, 32): {"ball_angle": 0.5},
        },
    )
    assert points[2].visible and points[2].reason == "motion_verified"
    assert all(not point.visible for point in points[3:])
    assert not any(operation == "update" for operation, _ in operations)


@pytest.mark.parametrize(
    "case",
    [
        "supported",
        "color_change",
        "no_overlap",
        "wrong_velocity",
        "low_cosine",
        "negative",
        "gap",
    ],
)
def test_small_ball_feature_change_needs_color_overlap_motion_and_negative_evidence(
    monkeypatch, tmp_path, case
):
    # Frame 499's actual ball had 115 pixels, local cosine .908 and color .936.
    # Such a modest feature change is not permission to accept an orange head:
    # all of the extra short-term evidence must still hold independently.
    y, x = np.indices((15, 13))
    ellipse_distance = ((x - 6) / 6) ** 2 + ((y - 7) / 7) ** 2
    shape = np.zeros((15, 13), bool)
    shape.flat[np.argsort(ellipse_distance.ravel(), kind="stable")[:115]] = True
    rows, cols = np.nonzero(shape)
    shape = shape[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]

    def ball_at(start):
        mask = np.zeros((64, 80), bool)
        mask[30 : 30 + shape.shape[0], start : start + shape.shape[1]] = shape
        return mask

    previous_x, current_x = 10, 12
    if case == "no_overlap":
        previous_x, current_x = 20, 34
    elif case == "wrong_velocity":
        previous_x, current_x = 20, 20
    cosine = 0.899 if case == "low_cosine" else 0.908
    color_similarity = 0.80 if case == "color_change" else 0.936
    color = np.zeros(64, np.float32)
    color[:2] = color_similarity**2, 1 - color_similarity**2
    ball_angle = float(np.arccos(cosine))
    current_index = 3 if case == "gap" else 2
    objects = [{8: 0}, {previous_x: 0}]
    if case == "gap":
        objects.append({})
    objects.append({current_x: 0.9})
    features = {
        (0, 8): {"mask": ball_at(8)},
        (1, previous_x): {"mask": ball_at(previous_x)},
        (current_index, current_x): {
            "mask": ball_at(current_x),
            "ball_angle": ball_angle,
            "color": color,
            "quality": 0.966,
        },
    }
    proposals = {current_index: [current_x]}
    if case == "negative":
        objects[0][60] = 0.9
        proposals[0] = [60]
        features[(0, 60)] = {"ball_angle": ball_angle, "color": color}
    points, operations, _, diagnostics = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=objects,
        video_predictions={1: ball_at(previous_x), current_index: ball_at(current_x)},
        proposals=proposals,
        features=features,
    )
    assert points[current_index].visible is (case == "supported")
    if case == "supported":
        assert points[current_index].source == "sam2"
        assert points[current_index].reason == "motion_verified"
    elif case == "negative":
        assert diagnostics["ball"]["negatives"] >= 1
    assert ("update", current_index) not in operations


@pytest.mark.parametrize("prior_observation", [False, True])
def test_ball_recovery_requires_recent_velocity_measurements(
    monkeypatch, tmp_path, prior_observation
):
    objects = [{8: 0}, {}, {8: 0.9}, {8: 0.9}]
    proposals = {2: [8], 3: [8]}
    if prior_observation:
        objects = [{8: 0}, {}, {}, {}, {}, {40: 0}, {}, {40: 0.9}]
        proposals = {5: [40], 7: [40]}
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=objects,
        video_predictions={},
        proposals=proposals,
    )
    if prior_observation:
        assert points[5].visible and points[5].source == "reacquired"
        assert all(not point.visible for point in points[6:])
    else:
        assert all(not point.visible for point in points[1:])


@pytest.mark.parametrize("subpixel_velocity", [False, True])
def test_ball_local_sam_search_can_recover_zero_velocity_after_global_miss(
    monkeypatch, tmp_path, subpixel_velocity
):
    # Two real observations establish zero (or half-pixel) velocity. Both the
    # video model and full-frame DINO retrieval miss the next ball observation.
    masks = []
    for width in (8, 9 if subpixel_velocity else 8, 10 if subpixel_velocity else 8):
        mask = np.zeros((64, 80), bool)
        mask[30:46, 8 : 8 + width] = True
        masks.append(mask)
    points, operations, searches, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=[{8: 0}, {8: 0}, {8: 0.9}],
        video_predictions={1: masks[1]},
        proposals={},
        features={
            (1, 8): {"mask": masks[1]},
            (2, 8): {"mask": masks[2], "quality": 0.839},
        },
    )
    assert any(index == 2 for index, _, _ in searches)
    assert ("segment", 2) in operations
    assert points[2].visible and points[2].source == "reacquired"
    assert points[2].reason == "motion_verified"
    assert points[2].bbox == (8, 30, 17 if subpixel_velocity else 15, 45)
    assert points[2].confidence == pytest.approx(0.839)
    assert ("update", 2) not in operations


@pytest.mark.parametrize("ball_first", [False, True])
def test_ball_continuity_candidate_survives_nested_hand_and_shirt_masks(
    monkeypatch, tmp_path, ball_first
):
    # Reproduces the frame-106 hierarchy: the actual 149-pixel ball has lower
    # whole-crop appearance than enclosing hand/shirt alternatives, but only
    # its shape, size and local ball vector agree with the recent ball track.
    large = np.zeros((64, 80), bool)
    large[22:54, 4:38] = True
    medium = np.zeros((64, 80), bool)
    medium[26:50, 10:30] = True
    ball = np.zeros((64, 80), bool)
    ball[30:45, 16:26] = True
    ball[30, 16] = False
    assert np.count_nonzero(ball) == 149
    candidates = [(large, 0.99), (medium, 0.98), (ball, 0.839)]
    if ball_first:
        candidates.reverse()
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        target="ball",
        objects=[{8: 0}, {12: 0}, {16: 0.9}],
        video_predictions={1: 12},
        proposals={2: [16]},
        features={
            (2, 4): {"angle": 0.1, "shape": (0.2, 0.2, 0.1, 0.4)},
            (2, 10): {"angle": 0.2, "shape": (0.3, 0.3, 0.2, 0.5)},
            (2, 16): {
                "mask": ball,
                "candidates": candidates,
                "ball_angle": float(np.arccos(0.943)),
            },
        },
    )
    assert points[2].visible and points[2].source == "reacquired"
    assert points[2].reason == "motion_verified"
    assert points[2].bbox == (16, 30, 25, 44)
    assert points[2].confidence == pytest.approx(0.839)
    assert ("update", 2) not in operations


@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("stationary", True),
        ("translated", True),
        ("replacement", False),
        ("distant_mask", False),
        ("occluded", False),
        ("textureless", False),
    ],
)
def test_pixel_flow_requires_the_same_visible_foreground_texture(scenario, expected):
    from basketeditor.tracker import _flow_support

    # Actual OpenCV flow, without replacing the optical-flow implementation.
    # Independent random foreground texture at the SAME location is particularly
    # important: forward/backward LK alone can form self-consistent false matches.
    rng = np.random.default_rng(281)
    texture = rng.integers(20, 236, (96, 80, 3), dtype=np.uint8)
    old = np.full((192, 256, 3), 30, np.uint8)
    old[40:136, 50:130] = texture
    old_mask = np.zeros(old.shape[:2], bool)
    old_mask[40:136, 50:130] = True
    new = np.full_like(old, 30)
    mask = np.zeros_like(old_mask)
    if scenario in {"stationary", "replacement", "textureless"}:
        new[40:136, 50:130] = (
            rng.integers(20, 236, texture.shape, dtype=np.uint8)
            if scenario == "replacement"
            else texture
        )
        mask[40:136, 50:130] = True
    else:
        new[45:141, 62:142] = texture
        mask[45:141, 62:142] = True
    if scenario == "distant_mask":
        mask[:] = False
        mask[45:141, 170:250] = True
    elif scenario == "occluded":
        new[45:141, 62:142] = 30
    elif scenario == "textureless":
        old[:] = new[:] = 80
    assert bool(_flow_support(old, new, old_mask, mask)) is expected


def test_audit_preserves_continuous_mask_stream_and_weak_flow_evidence(
    monkeypatch, tmp_path
):
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: angle} for angle in (0, 0.15, 0.30, 0.45, 0.60, 0.60)],
        video_predictions={index: 8 for index in range(1, 6)},
        proposals={2: [8], 4: [8]},
        partial_candidates=True,
        # The audit's half-body alternative scores higher than the original
        # continuous full mask; audit must not replace or reseed that full mask.
        features={(4, 8): {"angles_by_area": {64: 0.0}}},
        audit_interval=2,
        live_audits=(2, 4),
        flow_support={4: True, 5: True},
    )
    assert all(point.visible for point in points)
    assert [index for action, index in operations if action == "seed"] == [0]
    assert all(point.source == "sam2" for point in points[1:])
    for point in points[4:]:
        assert point.bbox == (8, 30, 15, 45)
        assert point.reason == "motion_verified"
        assert ("update", point.frame_idx) not in operations
    assert ("flow", (3, 4)) in operations
    assert ("flow", (4, 5)) in operations


@pytest.mark.parametrize("interruption", ["gap", "cut"])
def test_pixel_flow_cannot_upgrade_recovery_after_a_gap_or_cut(
    monkeypatch, tmp_path, interruption
):
    objects = [{8: angle} for angle in (0, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6)]
    if interruption == "gap":
        objects[4] = {}
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=objects,
        video_predictions={index: 8 for index in range(1, 4)},
        proposals={4: [8] if interruption == "cut" else [], 5: [8], 6: [8]},
        cuts=[4] if interruption == "cut" else [],
        flow_support={index: True for index in range(1, 7)},
    )
    assert points[3].visible and points[3].reason == "motion_verified"
    assert all(not point.visible for point in points[4:])
    assert all(pair[1] <= 3 for action, pair in operations if action == "flow")
    assert ("update", 3) not in operations


@pytest.mark.parametrize("later_pixel_support", [False, True])
def test_motion_frames_cannot_accumulate_gallery_trust_for_a_later_weak_identity(
    monkeypatch, tmp_path, later_pixel_support
):
    # Without resetting the trusted run, two motion-only frames plus one .929
    # appearance stored that last candidate. After a gap it matched its own
    # automatic template at 1.0 and bypassed the .94 global-recovery floor.
    points, operations, _, diagnostics = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[
            {8: 0, 60: 0.9},
            {8: 0.43},
            {8: 0.43},
            {8: -0.4},
            {},
            {8: -0.4},
        ],
        video_predictions={1: 8, 2: 8, 3: 8},
        proposals={0: [60], 5: [8]},
        flow_support={1: True, 2: True, 3: later_pixel_support},
    )
    assert all(point.reason == "motion_verified" for point in points[1:3])
    assert points[3].visible is later_pixel_support
    assert ("update", 3) not in operations
    assert diagnostics["player"]["templates"] == 0
    assert not points[4].visible
    assert not points[5].visible


@pytest.mark.parametrize("supported_by_pixels", [False, True])
def test_strong_dino_pose_change_requires_local_or_pixel_continuity(
    monkeypatch, tmp_path, supported_by_pixels
):
    # A .89 fixed-anchor match can describe a new pose or a similar teammate.
    # At the same location, unchanged mask geometry alone cannot choose between
    # them when adjacent appearance changes substantially (cosine about .88).
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {8: 0.5}, {8: 0.5}],
        video_predictions={1: 8, 2: 8},
        proposals={1: [8], 2: [8]},
        flow_support={1: supported_by_pixels, 2: supported_by_pixels},
    )
    assert all(point.visible is supported_by_pixels for point in points[1:])
    if supported_by_pixels:
        assert all(point.source == "sam2" for point in points[1:])
    else:
        # A failed pixel continuation is a genuine rejection, so later repeated
        # candidates must meet the full .94 recovery gate, not the .86 track gate.
        assert all(point.source != "reacquired" for point in points[1:])


def test_bbox_overlap_cannot_bypass_failed_foreground_mask_continuity(
    monkeypatch, tmp_path
):
    # Connected U and T silhouettes occupy nearly the same bounding box, while
    # fewer than 12% of their foreground pixels overlap. A rejected strong
    # observation must not enter through the weaker bbox-only fallback.
    previous = np.zeros((64, 80), bool)
    previous[30:46, 8:11] = True
    previous[30:46, 14:16] = True
    previous[30:33, 8:16] = True
    current = np.zeros_like(previous)
    current[33:46, 11:14] = True
    current[43:46, 8:16] = True
    assert (previous & current).sum() / (previous | current).sum() < 0.12
    points, operations, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {8: 0}],
        video_predictions={1: current},
        proposals={},
        groups={
            ("player", 0): [
                {"target": "player", "frame_idx": 0, "x": 9, "y": 38, "label": 1}
            ]
        },
        features={(0, 8): {"mask": previous}},
        flow_support={1: False},
    )
    assert not points[1].visible
    assert ("seed", 1) not in operations


@pytest.mark.parametrize(
    "appearance_score,may_confirm", [(0.91584, False), (0.96, True)]
)
def test_temporal_confirmation_resolves_margin_only_after_sufficient_identity_appearance(
    monkeypatch, tmp_path, appearance_score, may_confirm
):
    # Reproduces the natural-video #16 switch: score .91584 and margin .0516
    # stayed stable, but repeating that appearance did not prove it was #8.
    # A strong .96 appearance can still use time to resolve the same margin.
    best = float(np.arccos((appearance_score - 0.1) / 0.9))
    second = float(np.arccos((appearance_score - 0.0516 - 0.1) / 0.9))
    points, _, _, _ = run_identity_case(
        monkeypatch,
        tmp_path,
        objects=[{8: 0}, {}, *[{8: best, 60: second} for _ in range(4)]],
        video_predictions={4: 8, 5: 8},
        proposals={index: [8, 60] for index in range(2, 6)},
    )
    assert points[2].identity_score == pytest.approx(appearance_score, abs=1e-5)
    assert points[2].identity_margin == pytest.approx(0.0516, abs=1e-5)
    if may_confirm:
        assert not points[2].visible and points[2].source == "pending"
        assert points[3].visible and points[3].source == "reacquired"
        assert all(point.visible and point.center[0] < 20 for point in points[3:])
    else:
        assert all(not point.visible for point in points[1:])
