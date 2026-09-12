"""Identity safety invariants and whole-image/small-object recall contracts."""

from dataclasses import replace

import numpy as np
import pytest

from basketeditor.reid import (
    Appearance,
    DinoExtractor,
    IdentityMemory,
    load_memories,
    save_memories,
)


def descriptor(angle=0.0, axis=1, bbox=(10, 10, 30, 50)):
    vector = np.zeros(8, np.float32)
    vector[0], vector[axis] = np.cos(angle), np.sin(angle)
    color = np.zeros(64, np.float32)
    color[24] = 1
    return Appearance(
        vector,
        np.tile(vector, (3, 1)),
        color,
        np.tile(vector, (2, 1)),
        bbox,
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
    )


def test_manual_identity_survives_long_absence_and_rejects_impostor():
    memory = IdentityMemory("player")
    anchor = descriptor()
    memory.add_anchor(anchor)
    # No lifetime counter or TTL discards identity, even after many missing frames.
    for _ in range(400):
        assert memory.decide([], recovering=True).index is None
    result = memory.decide([(descriptor(np.pi / 2), 0.999), (anchor, 0.80)])
    assert result.index == 1 and result.strong
    assert memory.decide([(descriptor(np.pi / 2), 0.999)]).index is None


def test_similar_lookalikes_are_ambiguous_even_with_high_mask_quality():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    result = memory.decide([(descriptor(0.1), 0.99), (descriptor(0.11), 0.99)])
    assert result.index is None
    assert result.reason == "ambiguous_lookalikes"
    assert result.margin < 0.045


def test_only_high_appearance_with_uncertain_margin_is_confirmable():
    memory = IdentityMemory("ball")
    memory.add_anchor(descriptor())
    best = descriptor(np.arccos((0.96 - 0.1) / 0.9))
    runner = descriptor(np.arccos((0.91 - 0.1) / 0.9))
    decision = memory.decide([(best, 0.99), (runner, 0.99)])
    assert decision.index == 0 and not decision.strong
    assert decision.reason == "needs_confirmation"
    assert not memory.update(best, 100, verified=False)
    assert memory.gallery == []


def test_repeated_moderate_appearance_cannot_establish_lost_identity():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    # Real same-team #16 was accepted after two .916 frames although the
    # annotated #8 was elsewhere. Repeating appearance is not new ID evidence.
    teammate = descriptor(np.arccos((0.915843 - 0.1) / 0.9))
    for _ in range(10):
        decision = memory.decide([(teammate, 0.99)], recovering=True)
        assert decision.index is None and not decision.strong
    # Spatially verified, uninterrupted propagation has a separate contract.
    assert memory.decide([(teammate, 0.99)], recovering=False).strong


def test_parts_can_reject_global_semantic_match():
    memory = IdentityMemory("player")
    anchor = descriptor()
    memory.add_anchor(anchor)
    same_jersey_wrong_parts = replace(anchor, parts=descriptor(np.pi / 2).parts)
    assert memory.decide([(same_jersey_wrong_parts, 0.99)]).index is None


def test_stable_teammate_is_not_confirmable_without_negative_examples():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    teammate = descriptor(np.arccos((0.84 - 0.1) / 0.9))
    for _ in range(20):
        assert memory.decide([(teammate, 0.99)]).index is None
        assert memory.decide([(teammate, 0.99)], recovering=False).index is None


def test_known_same_uniform_distractor_requires_positive_negative_margin():
    memory = IdentityMemory("player")
    anchor, teammate = descriptor(), descriptor(0.3)
    memory.add_anchor(anchor)
    assert memory.add_negative(teammate)
    result = memory.decide([(teammate, 0.99)])
    assert result.index is None
    assert result.reason == "matches_known_distractor"
    assert memory.similarity(teammate).negative_margin < 0
    assert memory.decide([(anchor, 0.99)]).strong
    assert not memory.update(teammate, 500)


def test_disjoint_indistinguishable_object_is_retained_as_ambiguity():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    # The caller established that this matching outfit belongs to another body.
    assert memory.add_negative(replace(descriptor(), bbox=(80, 10, 100, 50)))
    result = memory.decide([(descriptor(), 0.99)])
    assert result.index is None and result.reason == "matches_known_distractor"


def test_clear_known_negative_does_not_hide_another_valid_identity():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    known, target = descriptor(0.34), descriptor(-0.345)
    memory.add_negative(known)
    assert memory.similarity(known).score > memory.similarity(target).score
    decision = memory.decide([(known, 0.99), (target, 0.99)])
    assert decision.index == 1 and decision.strong
    rejected = memory.decide([(known, 0.99)])
    assert rejected.index is None
    assert rejected.reason == "matches_known_distractor"
    assert rejected.score == pytest.approx(memory.similarity(known).score)


def test_uncertain_negative_remains_a_competing_lookalike():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    memory.add_negative(descriptor(0.34))
    uncertain = descriptor(0.31)
    match = memory.similarity(uncertain)
    assert match.negative_score > 0.98 and match.negative_margin > -0.05
    decision = memory.decide([(descriptor(), 0.99), (uncertain, 0.99)])
    assert decision.index is None and decision.reason == "ambiguous_lookalikes"


def test_candidate_below_positive_floor_still_counts_as_ambiguity():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    best = descriptor(np.arccos((0.870 - 0.1) / 0.9))
    uncertain = descriptor(np.arccos((0.859 - 0.1) / 0.9))
    decision = memory.decide([(best, 0.99), (uncertain, 0.99)], recovering=False)
    assert decision.index is None and decision.reason == "ambiguous_lookalikes"


def test_negative_capacity_keeps_new_hard_distractor_over_easy_background():
    def item(vector):
        base = descriptor()
        return replace(
            base,
            vector=vector,
            parts=np.tile(vector, (3, 1)),
            search_tokens=vector[None],
        )

    basis = np.eye(26, dtype=np.float32)
    memory = IdentityMemory("player")
    memory.add_anchor(item(basis[0]))
    for i in range(1, 25):
        assert memory.add_negative(item(basis[i]))
    near_clone = item(np.cos(0.2) * basis[0] + np.sin(0.2) * basis[25])
    assert memory.add_negative(near_clone)
    assert len(memory.negatives) == 24
    assert memory.decide([(near_clone, 0.99)]).index is None


def test_anchor_cannot_be_mutated_or_replaced_by_auto_updates():
    original = descriptor()
    memory = IdentityMemory("player", max_templates=2)
    memory.add_anchor(original)
    saved = memory.anchors[0].vector.copy()
    original.vector[:] = 0
    np.testing.assert_array_equal(memory.anchors[0].vector, saved)
    with pytest.raises(ValueError, match="read-only"):
        memory.anchors[0].vector[0] = 0
    for frame, axis in enumerate(range(1, 7), 1):
        assert memory.update(descriptor(0.4, axis=axis), frame * 10)
    assert len(memory.gallery) == 2
    assert len(memory.anchors) == 1
    np.testing.assert_array_equal(memory.anchors[0].vector, saved)
    assert not memory.update(descriptor(np.pi / 2), 1000)


def test_near_duplicate_views_do_not_fill_gallery():
    memory = IdentityMemory("hoop")
    memory.add_anchor(descriptor())
    assert not memory.update(descriptor(0.01), 100)
    assert memory.update(descriptor(0.4), 200)
    assert not memory.update(descriptor(0.4), 10000)
    assert len(memory.gallery) == 1


def test_invalid_sam_output_cannot_restore_identity():
    memory = IdentityMemory("ball")
    memory.add_anchor(descriptor())
    for quality in (float("inf"), float("nan")):
        result = memory.decide([(descriptor(), quality)])
        assert result.index is None
        assert result.reason == "poor_segmentation"


def test_invalid_mask_does_not_hide_another_valid_candidate():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    result = memory.decide([(descriptor(), float("nan")), (descriptor(0.3), 0.8)])
    assert result.index == 1 and result.strong


def test_low_predicted_mask_iou_does_not_override_strong_identity():
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor())
    # Real SAM2 multimask output gave a correct full person IoU=.978 while
    # predicting quality=.028. Identity and geometry must choose the object.
    result = memory.decide([(descriptor(0.2), 0.028), (descriptor(1.1), 0.99)])
    assert result.index == 0 and result.strong


def test_npz_preserves_separate_target_anchors_and_bounded_templates(tmp_path):
    player, ball = IdentityMemory("player", 2), IdentityMemory("ball")
    player.add_anchor(descriptor())
    player.add_anchor(descriptor(0.7))
    player.update(descriptor(0.35, axis=2), 20)
    player.add_negative(descriptor(np.pi / 2))
    ball.add_anchor(descriptor(np.pi / 2))
    path = tmp_path / "identities.npz"
    save_memories(path, {"player": player, "ball": ball})
    with np.load(path, allow_pickle=False) as archive:
        assert "metadata" in archive
    restored = load_memories(path)
    assert restored["player"].stats() == player.stats()
    assert restored["player"].max_templates == 2
    assert restored["player"].similarity(descriptor()) == player.similarity(
        descriptor()
    )
    assert restored["ball"].decide([(descriptor(), 0.99)]).index is None
    assert not restored["player"].anchors[0].vector.flags.writeable
    assert len(restored["player"].negatives) == 1


def test_ball_features_survive_npz_and_are_protected(tmp_path):
    base = descriptor()
    ball = replace(base, ball_vector=base.vector.copy(), shape=(0.9, 0.7, 0.8, 0.95))
    memory = IdentityMemory("ball")
    memory.add_anchor(ball)
    ball.ball_vector[:] = 0
    assert np.linalg.norm(memory.anchors[0].ball_vector) == pytest.approx(1)
    path = tmp_path / "ball.npz"
    save_memories(path, {"ball": memory})
    restored = load_memories(path)["ball"]
    assert restored.anchors[0].shape == ball.shape
    assert not restored.anchors[0].ball_vector.flags.writeable
    assert restored.ball_similarity(restored.anchors[0]).plausible


def test_ball_semantics_cannot_relax_strict_identity_or_player_threshold():
    shape = (0.9, 0.7, 0.8, 0.95)
    anchor = replace(descriptor(), ball_vector=descriptor().vector, shape=shape)
    # Real tiny heads also have ball-like shape and very similar neutral features.
    head = replace(descriptor(0.9), ball_vector=descriptor(0.1).vector, shape=shape)
    memory = IdentityMemory("ball")
    memory.add_anchor(anchor)
    assert memory.ball_similarity(head).plausible
    assert memory.decide([(head, 0.99)]).index is None
    player = IdentityMemory("player")
    player.add_anchor(anchor)
    assert not player.ball_similarity(head).plausible
    assert player.decide([(head, 0.99)]).index is None


def test_ball_shape_and_known_head_add_vetoes_for_motion_candidates():
    shape = (0.9, 0.7, 0.8, 0.95)
    anchor = replace(descriptor(), ball_vector=descriptor().vector, shape=shape)
    hand = replace(anchor, shape=(0.35, 0.6, 0.23, 0.75))
    head = replace(descriptor(0.9), ball_vector=descriptor(0.35).vector, shape=shape)
    memory = IdentityMemory("ball")
    memory.add_anchor(anchor)
    assert not memory.ball_similarity(hand).plausible
    assert memory.add_negative(head)
    assert not memory.ball_similarity(head).plausible
    assert memory.ball_similarity(anchor).plausible


def test_high_identity_score_cannot_turn_a_hand_into_a_ball():
    shape = (0.9, 0.7, 0.8, 0.95)
    anchor = replace(descriptor(), ball_vector=descriptor().vector, shape=shape)
    arm = replace(anchor, shape=(0.35, 0.6, 0.23, 0.75))
    memory = IdentityMemory("ball")
    memory.add_anchor(anchor)
    rejected = memory.decide([(arm, 0.99)])
    assert rejected.index is None and rejected.reason == "non_ball_shape"
    assert not memory.update(arm, 100)
    # Rejecting the higher-ranked arm does not hide a real round ball candidate.
    candidate = replace(descriptor(0.2), ball_vector=descriptor().vector, shape=shape)
    assert memory.decide([(arm, 0.99), (candidate, 0.8)]).index == 1


def test_neutral_ball_semantics_do_not_override_a_strong_strict_identity():
    shape = (0.9, 0.7, 0.8, 0.95)
    memory = IdentityMemory("ball")
    memory.add_anchor(
        replace(descriptor(), ball_vector=descriptor().vector, shape=shape)
    )
    observed = replace(descriptor(0.1), ball_vector=descriptor(1.2).vector, shape=shape)
    assert not memory.ball_similarity(observed).plausible
    assert memory.decide([(observed, 0.9)]).strong


def test_ball_semantics_cannot_override_a_known_original_appearance_negative():
    shape = (0.9, 0.7, 0.8, 0.95)
    memory = IdentityMemory("ball")
    anchor = replace(descriptor(), ball_vector=descriptor().vector, shape=shape)
    shoe = replace(
        descriptor(np.pi / 2), ball_vector=descriptor(np.pi / 2).vector, shape=shape
    )
    memory.add_anchor(anchor)
    memory.add_negative(shoe)
    # A semantic feature change can make a cropped shoe resemble a ball, while
    # its original appearance still clearly identifies the known non-target.
    candidate = replace(shoe, ball_vector=anchor.ball_vector)
    assert memory.ball_similarity(candidate).positive == pytest.approx(1)
    assert not memory.ball_similarity(candidate).plausible


def test_old_descriptors_have_no_implicit_ball_semantics():
    memory = IdentityMemory("ball")
    memory.add_anchor(descriptor())
    assert not memory.ball_similarity(descriptor()).plausible


class RecordingExtractor(DinoExtractor):
    """Instrument image coverage without downloading or invoking a backbone."""

    def __init__(self):
        self.image_size = 448
        self._image, self._maps, self.encoded = None, {}, []

    def _encode(self, image, size):
        self.encoded.append((image.shape, size))
        h, w = image.shape[:2]
        rows = max(1, round(h * size / max(h, w) / 16))
        cols = max(1, round(w * size / max(h, w) / 16))
        features = np.zeros((rows, cols, 8), np.float32)
        features[..., 0] = 1
        return features


def test_tiny_ball_is_actually_cropped_and_resized_before_describing():
    extractor = RecordingExtractor()
    frame = np.zeros((480, 640, 3), np.uint8)
    mask = np.zeros(frame.shape[:2], bool)
    mask[200:205, 300:305] = True
    appearance = extractor.describe(frame, mask)
    assert appearance.area == 25
    assert appearance.bbox == (300, 200, 305, 205)
    input_shape, requested_size = extractor.encoded[0]
    assert input_shape[0] < 20 and input_shape[1] < 20
    assert requested_size == 224
    assert np.isfinite(appearance.vector).all()
    assert np.linalg.norm(appearance.vector) == pytest.approx(1)
    assert appearance.parts.shape == (3, 8)
    assert appearance.color.sum() == pytest.approx(1)


def test_ball_descriptor_uses_neutral_background_and_four_rotations():
    extractor = RecordingExtractor()
    frame = np.full((64, 64, 3), 255, np.uint8)
    mask = np.zeros(frame.shape[:2], bool)
    mask[20:30, 30:40] = True
    appearance = extractor.describe(frame, mask, target="ball")
    assert appearance.ball_vector.shape == (8,)
    assert np.linalg.norm(appearance.ball_vector) == pytest.approx(1)
    assert appearance.shape[0] == 1
    assert appearance.shape[1] == 1
    assert len(extractor.encoded) == 6  # Object, scene context, four neutral views.
    assert all(
        shape == (15, 15, 3) and size == 224 for shape, size in extractor.encoded[-4:]
    )


def test_recovery_tiles_cover_entire_image_on_every_call():
    extractor = RecordingExtractor()
    frame = np.zeros((720, 1280, 3), np.uint8)
    memory = IdentityMemory("ball")
    memory.add_anchor(descriptor(bbox=(30, 30, 38, 38)))
    proposals = extractor.search(frame, memory, limit=6, tile_offset=99)
    coverage = np.zeros(frame.shape[:2], bool)
    for box in extractor._maps:
        if box != (0, 0, 1280, 720):
            x1, y1, x2, y2 = box
            coverage[y1:y2, x1:x2] = True
    assert coverage.all()
    assert 2 <= len(proposals) <= 6
    assert all(0 <= p.point[0] < 1280 and 0 <= p.point[1] < 720 for p in proposals)
    # One full frame, at most nine covering tiles, and five precise local crops.
    assert len(extractor._maps) <= 15
    count = len(extractor.encoded)
    extractor.search(frame, memory)
    assert len(extractor.encoded) == count


def test_feature_cache_discards_previous_frame():
    extractor = RecordingExtractor()
    memory = IdentityMemory("player")
    memory.add_anchor(descriptor(bbox=(100, 80, 200, 380)))
    for _ in range(4):
        frame = np.zeros((480, 640, 3), np.uint8)
        extractor.search(frame, memory)
        assert extractor._image is frame
        assert len(extractor._maps) == 1


def test_empty_or_missized_mask_cannot_create_identity():
    extractor = RecordingExtractor()
    frame = np.zeros((32, 32, 3), np.uint8)
    assert extractor.describe(frame, np.zeros((32, 32))) is None
    with pytest.raises(ValueError, match="dimensions"):
        extractor.describe(frame, np.zeros((16, 16)))
