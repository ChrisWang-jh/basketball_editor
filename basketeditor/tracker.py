"""Identity-checked SAM2 propagation and independent full-frame recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .models import TARGET_IDS, TrackPoint


class Segmenter:
    """Image prompts share model weights, never the video's temporal memory."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.key = None

    def predict(self, rgb, frame_idx, *, points=None, proposal=None, small=False):
        from .sam2 import mask_to_track_point, select_component

        h, w = rgb.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
        if proposal is not None and small:
            bx0, by0, bx1, by1 = proposal.box
            cx, cy = proposal.point
            radius = max(64, int(max(bx1 - bx0, by1 - by0) * 2))
            x0, y0 = max(0, int(cx) - radius), max(0, int(cy) - radius)
            x1, y1 = min(w, int(cx) + radius), min(h, int(cy) + radius)
        elif small and points and not any("box" in p for p in points):
            # A 15-pixel ball is otherwise only a few SAM encoder pixels. Use
            # the same high-resolution crop for manual anchors and recovery;
            # retain every positive/negative click inside the crop.
            xs, ys = [p["x"] for p in points], [p["y"] for p in points]
            padding = max(64, round(max(max(xs) - min(xs), max(ys) - min(ys))))
            x0, y0 = max(0, int(min(xs)) - padding), max(0, int(min(ys)) - padding)
            x1, y1 = (
                min(w, int(max(xs)) + padding + 1),
                min(h, int(max(ys)) + padding + 1),
            )
        key = (frame_idx, x0, y0, x1, y1)
        if self.key != key:
            self.predictor.set_image(rgb[y0:y1, x0:x1])
            self.key = key
        box = None
        if proposal is not None:
            coords = np.asarray([proposal.point], np.float32) - [x0, y0]
            labels = np.ones(1, np.int32)
        else:
            coords = np.asarray(
                [[p["x"] - x0, p["y"] - y0] for p in points if "box" not in p],
                np.float32,
            ).reshape(-1, 2)
            labels = np.asarray(
                [p["label"] for p in points if "box" not in p], np.int32
            )
            boxes = [p["box"] for p in points if "box" in p]
            if boxes:
                box = np.asarray(boxes[-1], np.float32) - [x0, y0, x0, y0]
        masks, scores, _ = self.predictor.predict(
            point_coords=coords if len(coords) else None,
            point_labels=labels if len(labels) else None,
            box=box,
            multimask_output=box is None and len(coords) == 1,
            return_logits=True,
        )
        result = []
        for index in np.argsort(np.asarray(scores).reshape(-1))[::-1]:
            logits = np.squeeze(masks[index])
            local = logits > 0
            if np.count_nonzero(local) < 4:
                continue
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = local
            prompts = (
                [{"x": proposal.point[0], "y": proposal.point[1], "label": 1}]
                if proposal is not None
                else points
            )
            mask = (
                select_component(np.where(mask, 16.0, -16.0), np, cv2, prompts or [])
                > 0
            )
            # Predicted IoU ranks alternatives but may be near zero for a correct
            # full-body mask. Report the actual foreground logit confidence;
            # DINO independently determines whether this is the target identity.
            selected_logits = np.where(mask[y0:y1, x0:x1], logits, -16.0)
            quality = mask_to_track_point(frame_idx, selected_logits, np).confidence
            result.append((mask, quality))
        return result


def _mask_iou(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / max(1, union)


def _same_object(a, b):
    intersection = np.count_nonzero(a & b)
    return (
        _mask_iou(a, b) > 0.65
        or intersection / max(1, min(np.count_nonzero(a), np.count_nonzero(b))) > 0.85
    )


def _box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(1, union - intersection)


def _geometry_ok(desc, memory, previous, info, *, recovering=False):
    x0, y0, x1, y1 = desc.bbox
    if not (0 <= x0 < x1 <= info.width and 0 <= y0 < y1 <= info.height):
        return False
    if desc.area < 4 or desc.area > info.width * info.height * 0.75:
        return False
    areas = [a.area for a in memory.anchors]
    if areas and not min(areas) * 0.025 <= desc.area <= max(areas) * 20:
        return False
    if previous is not None and not recovering:
        center = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2])
        speed = (
            np.hypot(info.width, info.height)
            * {"player": 1.0, "hoop": 0.8, "ball": 4.0}[memory.target]
            / info.fps
        )
        if (
            np.linalg.norm(center - previous.center)
            > speed + max(x1 - x0, y1 - y0) * 0.5
        ):
            return False
    return True


def _motion_prediction(history, index, info):
    """Two recent observations define velocity, including a stationary object."""
    if len(history) < 2:
        return None
    recent, earlier = history[-1][0], history[-2][0]
    dt, gap = recent.frame_idx - earlier.frame_idx, index - recent.frame_idx
    window = max(1, round(info.fps * 0.12))
    if not (0 < dt <= window and 0 < gap <= window):
        return None
    center = np.asarray(recent.center, dtype=float)
    return center + (center - earlier.center) * gap / dt


def _flow_support(previous_rgb, rgb, previous_mask, mask):
    """Verify that foreground pixels, rather than a similar object, continued.

    Forward/backward LK rejects occluded or ambiguous points. A small boundary
    tolerance allows SAM's contour to move without borrowing background points.
    This is local motion evidence, never a descriptor for global reacquisition.
    """
    if previous_rgb is None or rgb is None or previous_mask.shape != mask.shape:
        return False
    old_gray = cv2.cvtColor(previous_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    interior = cv2.erode(previous_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    points = cv2.goodFeaturesToTrack(
        old_gray, maxCorners=80, qualityLevel=0.01, minDistance=5, mask=interior
    )
    if points is None or len(points) < 12:
        return False
    params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    forward, status, _ = cv2.calcOpticalFlowPyrLK(
        old_gray, gray, points, None, **params
    )
    if forward is None or status is None:
        return False
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(
        gray, old_gray, forward, None, **params
    )
    if backward is None or back_status is None:
        return False
    coordinates = forward[:, 0]
    valid = (
        (status[:, 0] > 0)
        & (back_status[:, 0] > 0)
        & np.isfinite(coordinates).all(axis=1)
        & (np.linalg.norm(backward[:, 0] - points[:, 0], axis=1) < 1.5)
    )
    x, y = (
        np.rint(np.nan_to_num(coordinates, nan=-1, posinf=-1, neginf=-1))
        .astype(np.int64)
        .T
    )
    height, width = mask.shape
    valid &= (x >= 0) & (x < width) & (y >= 0) & (y < height)
    # LK can close a forward/backward loop on unrelated texture at the same
    # location. Check local, contrast-normalized appearance as well as motion.
    for i in np.flatnonzero(valid):
        before = cv2.getRectSubPix(old_gray, (7, 7), tuple(points[i, 0])).astype(float)
        after = cv2.getRectSubPix(gray, (7, 7), tuple(coordinates[i])).astype(float)
        before -= before.mean()
        after -= after.mean()
        correlation = float((before * after).sum()) / max(
            1e-8, float(np.linalg.norm(before) * np.linalg.norm(after))
        )
        valid[i] = correlation >= 0.6
    count = int(valid.sum())
    if count < 12 or count < len(points) * 0.25:
        return False
    boundary = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    return float(boundary[y[valid], x[valid]].mean()) >= 0.8


def _continuity_ok(
    memory, candidate, history, index, info, *, previous_rgb=None, rgb=None
):
    """Additional evidence for an unbroken track, never a new global identity."""
    if not history:
        return False
    mask, quality, desc = candidate
    point, old, old_mask = history[-1]
    gap = index - point.frame_idx
    ball = memory.target == "ball"
    if not 1 <= gap <= (max(1, round(info.fps * 0.12)) if ball else 1):
        return False
    if quality < 0.65 or not _geometry_ok(desc, memory, None if ball else point, info):
        return False
    ratio = desc.area / max(1, old.area)
    if not 0.5 <= ratio <= 2.0:
        return False
    if ball:
        evidence = memory.ball_similarity(desc)
        if not evidence.plausible or old.ball_vector is None:
            return False
        adjacent_ball = float(desc.ball_vector @ old.ball_vector)
        if adjacent_ball < 0.90:
            return False
        color = float(np.sqrt(desc.color * old.color).sum())
        if color < 0.60:
            return False
        center = np.asarray(
            [(desc.bbox[0] + desc.bbox[2]) / 2, (desc.bbox[1] + desc.bbox[3]) / 2]
        )
        predicted = _motion_prediction(history, index, info)
        if predicted is None:
            predicted = np.asarray(point.center, dtype=float)
        radius = max(
            10, max(desc.bbox[2] - desc.bbox[0], desc.bbox[3] - desc.bbox[1]) * 1.4
        )
        residual = float(np.linalg.norm(center - predicted))
        if adjacent_ball < 0.92:
            # A hand can expose a different fraction of the same tiny ball.
            # Modest feature variation is acceptable only with overlapping
            # observed masks, stable color and a much tighter motion residual.
            diameter = max(desc.bbox[2] - desc.bbox[0], desc.bbox[3] - desc.bbox[1])
            if (
                gap != 1
                or color < 0.85
                or _mask_iou(mask, old_mask) < 0.4
                or residual > max(4, diameter * 0.5)
            ):
                return False
        return residual <= radius
    if memory.target != "player" or _box_iou(desc.bbox, old.bbox) < 0.35:
        return False
    match = memory.similarity(desc)
    adjacent = float(desc.vector @ old.vector)
    if (
        match.anchor_score >= 0.86
        and match.anchor_part_score >= 0.72
        and match.negative_margin >= 0.0
        and match.negative_score < 0.985
        and adjacent >= 0.95
        and _mask_iou(mask, old_mask) >= 0.35
    ):
        return True
    # A turn can change whole-body DINO features while the same foreground
    # pixels continue moving. Fixed anchors still constrain an unbroken track.
    appearance = (
        match.anchor_score >= 0.80
        and match.anchor_part_score >= 0.72
        and match.negative_margin >= -0.06
        and adjacent >= 0.90
    )
    # During an observed exit the image contains a shrinking fraction of the
    # person. Require the same border, overlap and pixel motion, not a full-body
    # match against a fragment. This path cannot be used by a new entrant.
    edges = (0, 0, info.width, info.height)
    exiting = (
        any(
            abs(a - edge) <= 1 and abs(b - edge) <= 1
            for a, b, edge in zip(old.bbox, desc.bbox, edges)
        )
        and ratio <= 1.05
        and _mask_iou(mask, old_mask) >= 0.5
        and match.anchor_score >= 0.60
        and match.anchor_part_score >= 0.55
        and adjacent >= 0.85
    )
    return (
        match.negative_score < 0.985
        and (appearance or exiting)
        and _flow_support(previous_rgb, rgb, old_mask, mask)
    )


def _prune_state(state, frame_idx, keep=32):
    """Only recent temporal tensors survive; permanent identity is kept separately."""
    for output in state.get("output_dict_per_obj", {}).values():
        bank = output.get("non_cond_frame_outputs", {})
        for i in list(bank):
            if i < frame_idx - keep:
                bank.pop(i)
    for bank in state.get("frames_tracked_per_obj", {}).values():
        for i in list(bank):
            if i < frame_idx - keep:
                bank.pop(i)


def track_identities(
    predictor,
    image_predictor,
    state,
    frames_dir,
    info,
    groups,
    options,
    masks_dir,
    scene_cuts,
    progress,
    diagnostics,
    identity_file,
):
    from .reid import DinoExtractor

    extractor = DinoExtractor(
        Path(options.repo),
        Path(options.checkpoint),
        str(predictor.device),
        image_size=options.image_size,
    )
    try:
        return _track_identities(
            extractor,
            predictor,
            image_predictor,
            state,
            frames_dir,
            info,
            groups,
            options,
            masks_dir,
            scene_cuts,
            progress,
            diagnostics,
            identity_file,
        )
    finally:
        extractor.release()
        image_predictor.reset_predictor()


def _track_identities(
    extractor,
    predictor,
    image_predictor,
    state,
    frames_dir,
    info,
    groups,
    options,
    masks_dir,
    scene_cuts,
    progress,
    diagnostics,
    identity_file,
):
    from .reid import IdentityMemory, Proposal, save_memories
    from .sam2 import mask_to_track_point, select_component

    segmenter = Segmenter(image_predictor)
    memories = {name: IdentityMemory(name) for name in TARGET_IDS}
    anchors = {name: {} for name in TARGET_IDS}
    tracks = {
        name: [TrackPoint(i, source="lost") for i in range(info.frame_count)]
        for name in TARGET_IDS
    }

    def read(index):
        frame = cv2.imread(str(frames_dir / f"{index:08d}.jpg"))
        if frame is None:
            raise RuntimeError(f"Cannot decode tracking frame {index}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Every manual anchor is segmented independently; no automatic mask can enter
    # this immutable bank. Later annotations also identify earlier video frames.
    for (target, index), points in sorted(groups.items(), key=lambda item: item[0][1]):
        if not any(p["label"] == 1 for p in points):
            continue
        rgb = read(index)
        masks = segmenter.predict(rgb, index, points=points, small=target == "ball")
        if not masks:
            raise ValueError(
                f"{target} 第 {index} 帧的提示没有分割出目标，请调整提示点。"
            )
        mask, quality = masks[0]
        mask = select_component(np.where(mask, 16.0, -16.0), np, cv2, points) > 0
        desc = extractor.describe(rgb, mask, target=target)
        if desc is None:
            raise ValueError(f"{target} 第 {index} 帧没有足够的可见像素用于身份记忆。")
        memories[target].add_anchor(desc)
        anchors[target][index] = (mask, quality, desc)

    # Disjoint objects in a labelled frame supply fixed, known non-target views.
    for target, labelled in anchors.items():
        memory = memories[target]
        for index, (positive, _, positive_desc) in labelled.items():
            rgb = read(index)
            radius = max(3, min(15, round(np.sqrt(positive_desc.area) * 0.06)))
            protected = (
                cv2.dilate(
                    positive.astype(np.uint8),
                    np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8),
                )
                > 0
            )
            for proposal in extractor.search(rgb, memory, limit=12):
                for mask, quality in segmenter.predict(
                    rgb, index, proposal=proposal, small=target in {"ball", "hoop"}
                ):
                    overlap = np.count_nonzero(mask & protected) / max(
                        1, min(np.count_nonzero(mask), np.count_nonzero(positive))
                    )
                    if quality < 0.65 or overlap > 0.05:
                        continue
                    desc = extractor.describe(rgb, mask, target=target)
                    if desc is not None and _geometry_ok(
                        desc, memory, None, info, recovering=True
                    ):
                        memory.add_negative(desc)

    cuts = set(scene_cuts or [])
    total = max(1, sum(bool(a) for a in anchors.values()) * info.frame_count)
    completed = 0
    for target, oid in TARGET_IDS.items():
        memory = memories[target]
        if not anchors[target]:
            continue
        if masks_dir is not None:
            (masks_dir / target).mkdir(parents=True, exist_ok=True)
        stream = None
        previous = None
        previous_rgb = None
        history = []
        pending = None
        trusted_run = 0
        resets = 0
        searches = 0
        search_reports = []

        def stop():
            nonlocal stream, trusted_run, resets
            if stream is not None:
                stream.close()
                stream = None
            predictor.reset_state(state)
            trusted_run = 0
            resets += 1

        def seed(index, mask):
            nonlocal stream
            # Closing the generator precedes reset: suspended generators retain
            # references to old object dictionaries even after reset_state().
            stop()
            predictor.add_new_mask(state, index, oid, mask)
            stream = predictor.propagate_in_video(
                state,
                start_frame_idx=index,
                max_frame_num_to_track=info.frame_count - index - 1,
            )

        def corrected(rgb, index, mask, quality):
            negatives = [p for p in groups.get((target, index), []) if p["label"] == 0]
            if not negatives or not np.any(mask):
                return mask, quality
            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
            y, x = np.unravel_index(int(distance.argmax()), mask.shape)
            points = [{"x": float(x), "y": float(y), "label": 1}, *negatives]
            results = segmenter.predict(
                rgb, index, points=points, small=target == "ball"
            )
            for corrected_mask, corrected_quality in results:
                if all(
                    not corrected_mask[
                        min(mask.shape[0] - 1, round(p["y"])),
                        min(mask.shape[1] - 1, round(p["x"])),
                    ]
                    for p in negatives
                ):
                    return corrected_mask, corrected_quality
            return np.zeros_like(mask), 0.0

        def recover(rgb, index, current=None):
            nonlocal searches
            searches += 1
            candidates = [] if current is None else [current]
            proposals = extractor.search(
                rgb,
                memory,
                limit=options.search_candidates,
                previous_bbox=previous.bbox if previous is not None else None,
            )
            predicted = (
                _motion_prediction(history, index, info) if target == "ball" else None
            )
            if predicted is not None and (
                0 <= predicted[0] < info.width and 0 <= predicted[1] < info.height
            ):
                last = history[-1][0]
                shift = predicted - np.asarray(last.center)
                box = np.asarray(last.bbox) + np.tile(shift, 2)
                # Dense retrieval may prefer a chest or head over an 8-pixel ball.
                # This proposes a location only; the independent SAM2 mask must
                # still pass the same appearance, shape and motion evidence.
                proposals = [
                    Proposal(tuple(predicted), tuple(box), 0.0, "motion"),
                    *proposals,
                ]

            def rank(candidate):
                return (
                    target == "ball"
                    and _continuity_ok(memory, candidate, history, index, info),
                    memory.similarity(candidate[2]).score,
                )

            for proposal in proposals:
                for mask, quality in segmenter.predict(
                    rgb,
                    index,
                    proposal=proposal,
                    small=target in {"ball", "hoop"},
                ):
                    if quality < 0.45:
                        continue
                    mask, quality = corrected(rgb, index, mask, quality)
                    desc = extractor.describe(rgb, mask, target=target)
                    if desc is None or not _geometry_ok(
                        desc, memory, None, info, recovering=True
                    ):
                        continue
                    candidate = (mask, quality, desc)
                    duplicate = next(
                        (
                            j
                            for j, item in enumerate(candidates)
                            if _same_object(mask, item[0])
                        ),
                        None,
                    )
                    if duplicate is None:
                        candidates.append(candidate)
                    elif rank(candidate) > rank(candidates[duplicate]):
                        candidates[duplicate] = candidate
            selection = memory.decide(
                [(c[2], c[1]) for c in candidates], recovering=True
            )
            if len(search_reports) < 64:
                search_reports.append(
                    {
                        "frame": index,
                        "decision": selection.reason,
                        "candidates": [
                            {
                                "bbox": c[2].bbox,
                                "segmentation_quality": c[1],
                                "identity_score": memory.similarity(c[2]).score,
                            }
                            for c in candidates
                        ],
                    }
                )
            return candidates, selection

        stop()
        try:
            for index in range(info.frame_count):
                rgb = read(index)
                accepted = None
                source = "lost"
                reason = "no_verified_candidate"
                score = margin = None
                if index in cuts:
                    stop()
                    previous = pending = None
                    previous_rgb = None
                    history.clear()
                if index in anchors[target]:
                    accepted = anchors[target][index]
                    source, score, margin = "anchor", 1.0, 1.0
                    reason = "manual_anchor"
                    seed(index, accepted[0])
                else:
                    current = None
                    changed_mask = False
                    if stream is not None:
                        try:
                            output = next(stream)
                            while int(output[0]) < index:
                                output = next(stream)
                            raw_mask = output[2][list(output[1]).index(oid)]
                            raw_array = (
                                raw_mask.detach().float().cpu().numpy()
                                if hasattr(raw_mask, "detach")
                                else np.asarray(raw_mask)
                            )
                            mask = select_component(raw_mask, np, cv2, [], previous) > 0
                            raw_point = mask_to_track_point(
                                index,
                                np.where(mask, np.squeeze(raw_array), -16.0),
                                np,
                            )
                            changed_mask = np.count_nonzero(
                                raw_array > 0
                            ) - np.count_nonzero(mask) > max(
                                16, np.count_nonzero(mask) * 0.05
                            )
                            mask, quality = corrected(
                                rgb, index, mask, raw_point.confidence
                            )
                            changed_mask |= bool(groups.get((target, index)))
                            desc = extractor.describe(rgb, mask, target=target)
                            if desc is not None:
                                current = (mask, quality, desc)
                        except StopIteration:
                            stream = None
                    decision = memory.decide(
                        [] if current is None else [(current[2], current[1])],
                        recovering=False,
                    )
                    if (
                        current is not None
                        and decision.index is not None
                        and decision.strong
                        and _geometry_ok(current[2], memory, previous, info)
                        and (
                            target != "player"
                            or not history
                            or (
                                _mask_iou(current[0], history[-1][2]) >= 0.35
                                and float(current[2].vector @ history[-1][1].vector)
                                >= 0.95
                            )
                            or _flow_support(
                                previous_rgb, rgb, history[-1][2], current[0]
                            )
                        )
                    ):
                        accepted = current
                        source, score, margin = "sam2", decision.score, decision.margin
                        reason = "appearance_verified"
                        if changed_mask:
                            seed(index, current[0])
                    elif current is not None and _continuity_ok(
                        memory,
                        current,
                        history,
                        index,
                        info,
                        previous_rgb=previous_rgb,
                        rgb=rgb,
                    ):
                        accepted = current
                        source, reason = "sam2", "motion_verified"
                        score = memory.similarity(current[2]).score
                        margin = memory.similarity(current[2]).negative_margin
                        if changed_mask:
                            seed(index, current[0])
                    # Periodic competing-candidate checks catch confident switches
                    # that the temporal model itself fails to report as disappearance.
                    audit = accepted is not None and index % options.audit_interval == 0
                    if accepted is None or audit:
                        verified_current = accepted if audit else None
                        verified_reason = reason
                        if verified_current is None:
                            stop()
                        accepted = None
                        candidates, decision = recover(rgb, index, verified_current)
                        score, margin, reason = (
                            decision.score,
                            decision.margin,
                            decision.reason,
                        )
                        same_track = (
                            verified_current is not None
                            and decision.index is not None
                            and _same_object(
                                candidates[decision.index][0], verified_current[0]
                            )
                            and _geometry_ok(
                                candidates[decision.index][2], memory, previous, info
                            )
                        )
                        local_ball = [
                            c
                            for c in candidates
                            if target == "ball"
                            and _motion_prediction(history, index, info) is not None
                            and _continuity_ok(memory, c, history, index, info)
                        ]
                        if verified_current is not None:
                            # A distant look-alike scoring higher is not evidence
                            # that a still-verified continuous identity changed.
                            # Only a failed current track may reopen global identity
                            # assignment. Do not learn from this contested frame.
                            accepted = verified_current
                            source = "sam2"
                            reason = (
                                "motion_verified"
                                if verified_reason == "motion_verified"
                                else "appearance_audited"
                                if same_track
                                else "audit_competitor"
                            )
                            score = memory.similarity(accepted[2]).score
                            if not same_track:
                                margin = 0.0
                        elif decision.index is None and len(local_ball) == 1:
                            # Only a short, spatially continuous ball observation
                            # may use rotation-tolerant appearance. A head elsewhere
                            # or a ball after a long absence cannot use this path.
                            accepted = local_ball[0]
                            source, reason = "reacquired", "motion_verified"
                            score = memory.ball_similarity(accepted[2]).score
                            seed(index, accepted[0])
                        elif decision.index is not None:
                            candidate = candidates[decision.index]
                            count = 1
                            if pending is not None and pending[0] == index - 1:
                                old = pending[1]
                                new = candidate[2]
                                distance = np.linalg.norm(
                                    np.asarray(old.bbox[:2]) - np.asarray(new.bbox[:2])
                                )
                                size = max(
                                    new.bbox[2] - new.bbox[0], new.bbox[3] - new.bbox[1]
                                )
                                if float(
                                    old.vector @ new.vector
                                ) >= 0.85 and distance <= max(24, size):
                                    count = pending[2] + 1
                            if decision.strong or (
                                count >= options.confirm_frames
                                and decision.margin >= 0.04
                            ):
                                accepted = candidate
                                source = "sam2" if same_track else "reacquired"
                                if same_track:
                                    reason = "appearance_audited"
                                seed(index, accepted[0])
                            else:
                                source = "pending"
                                pending = (index, candidate[2], count)
                        else:
                            source = (
                                "identity_rejected" if current is not None else "lost"
                            )
                            pending = None
                if accepted is not None:
                    mask, quality, desc = accepted
                    point = mask_to_track_point(index, np.where(mask, 16.0, -16.0), np)
                    point = replace(
                        point,
                        confidence=float(np.clip(quality, 0, 1)),
                        source=source,
                        anchored=source == "anchor",
                        identity_score=score,
                        identity_margin=margin,
                        reason=reason,
                    )
                    previous = point
                    previous_rgb = rgb
                    history.append((point, desc, mask))
                    history = history[-3:]
                    pending = None
                    trusted_run = (
                        0
                        if reason in {"audit_competitor", "motion_verified"}
                        else trusted_run + 1
                    )
                    if (
                        trusted_run >= 3
                        and source == "sam2"
                        and reason not in {"audit_competitor", "motion_verified"}
                    ):
                        memory.update(desc, index, verified=True)
                    if masks_dir is not None:
                        path = masks_dir / target / f"{index:08d}.png"
                        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
                            raise RuntimeError(f"Could not write mask: {path}")
                else:
                    point = TrackPoint(
                        index,
                        source=source,
                        identity_score=score,
                        identity_margin=margin,
                        reason=reason,
                    )
                tracks[target][index] = point
                _prune_state(state, index)
                completed += 1
                if progress and (completed % 5 == 0 or source == "reacquired"):
                    progress(
                        min(0.73, 0.20 + 0.53 * completed / total),
                        desc=f"身份追踪 {target} · {index + 1}/{info.frame_count} · {source}",
                    )
        finally:
            stop()
        diagnostics[target] = {
            **memory.stats(),
            "searches": searches,
            "memory_resets": resets,
            "search_samples": search_reports,
        }
    if identity_file is not None:
        save_memories(identity_file, memories)
    return tracks
