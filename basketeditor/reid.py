"""Local DINOv3 retrieval and protected, non-expiring target appearance memory.

Retrieval proposes locations, never identities. SAM2 must segment each proposal
before ``decide`` compares it with the immutable user anchors. Similarities are
uncalibrated scores, not probabilities; a look-alike may remain ambiguous.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def _unit(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-8)


def _ball_shape_valid(shape: tuple) -> bool:
    aspect, fill, circularity, solidity = shape
    return aspect >= 0.45 and fill >= 0.35 and circularity >= 0.30 and solidity >= 0.65


@dataclass(frozen=True)
class Appearance:
    vector: np.ndarray
    parts: np.ndarray
    color: np.ndarray
    search_tokens: np.ndarray
    bbox: tuple[int, int, int, int]
    area: int
    ball_vector: np.ndarray | None = None
    # Aspect ratio, bounding-box occupancy, circularity, and convex solidity.
    shape: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class BallMatch:
    """Semantic evidence for a motion-gated ball candidate, never an identity."""

    positive: float
    negative: float
    shape_score: float
    color_score: float
    score: float
    negative_margin: float
    plausible: bool


@dataclass(frozen=True)
class Match:
    score: float
    anchor_score: float
    part_score: float
    color_score: float
    anchor_part_score: float
    negative_score: float = 0.0
    negative_margin: float = 1.0


@dataclass(frozen=True)
class Selection:
    # A weak selection is a candidate for temporal confirmation, not a recovery.
    index: int | None
    strong: bool
    score: float
    margin: float
    reason: str


@dataclass(frozen=True)
class Proposal:
    point: tuple[float, float]
    box: tuple[float, float, float, float]
    similarity: float
    source: str = "dense"


def _protected(descriptor: Appearance) -> Appearance:
    arrays = []
    for value in (
        descriptor.vector,
        descriptor.parts,
        descriptor.color,
        descriptor.search_tokens,
    ):
        array = np.array(value, dtype=np.float32, copy=True)
        if not np.isfinite(array).all():
            raise ValueError("An appearance descriptor contains non-finite values")
        array.setflags(write=False)
        arrays.append(array)
    ball = None
    if descriptor.ball_vector is not None:
        ball = np.array(descriptor.ball_vector, dtype=np.float32, copy=True)
        if not np.isfinite(ball).all():
            raise ValueError("A ball descriptor contains non-finite values")
        ball.setflags(write=False)
    shape = (
        tuple(float(v) for v in descriptor.shape)
        if descriptor.shape is not None
        else None
    )
    if shape is not None and (
        len(shape) != 4 or not all(math.isfinite(v) for v in shape)
    ):
        raise ValueError("Invalid ball shape descriptor")
    return Appearance(
        *arrays, tuple(descriptor.bbox), int(descriptor.area), ball, shape
    )


class IdentityMemory:
    """Explicit user anchors plus a small, diverse gallery; there is no TTL.

    Automatic updates cannot replace an anchor, and must still match an anchor.
    The caller must additionally establish temporal consistency before updating.
    """

    def __init__(self, target: str, max_templates: int = 6):
        if target not in {"player", "ball", "hoop"}:
            raise ValueError(f"Unknown tracking target: {target}")
        if max_templates < 0:
            raise ValueError("max_templates must be non-negative")
        self.target, self.max_templates = target, max_templates
        self.anchors: list[Appearance] = []
        self.gallery: list[tuple[int, Appearance]] = []
        self.negatives: list[Appearance] = []

    def add_anchor(self, descriptor: Appearance) -> None:
        self.anchors.append(_protected(descriptor))

    def add_negative(self, descriptor: Appearance) -> bool:
        """Record a spatially disjoint object from a user-annotated frame.

        Only the caller can establish mask disjointness. Suspect recovered tracks
        must never become negatives. A disjoint look-alike is especially useful:
        retaining it lets the matcher express that the identity is ambiguous.
        """
        if not self.anchors:
            return False
        if (
            self.negatives
            and max(self._compare(descriptor, item)[0] for item in self.negatives)
            > 0.97
        ):
            return False
        if len(self.negatives) >= 24:

            def hardness(item):
                score = max(self._compare(item, anchor)[0] for anchor in self.anchors)
                if self.target == "ball" and item.ball_vector is not None:
                    score = max(
                        [
                            score,
                            *(
                                float(item.ball_vector @ anchor.ball_vector)
                                for anchor in self.anchors
                                if anchor.ball_vector is not None
                            ),
                        ]
                    )
                return score

            easiest = min(
                range(len(self.negatives)), key=lambda i: hardness(self.negatives[i])
            )
            if hardness(descriptor) <= hardness(self.negatives[easiest]) + 0.005:
                return False
            self.negatives[easiest] = _protected(descriptor)
        else:
            self.negatives.append(_protected(descriptor))
        return True

    @staticmethod
    def _compare(left: Appearance, right: Appearance) -> tuple[float, float, float]:
        whole = float(np.clip(left.vector @ right.vector, 0, 1))
        local = np.clip(left.parts @ right.parts.T, 0, 1)
        parts = float(0.7 * np.diag(local).mean() + 0.3 * local.max(axis=1).mean())
        color = float(np.clip(np.sqrt(left.color * right.color).sum(), 0, 1))
        return 0.65 * whole + 0.25 * parts + 0.10 * color, parts, color

    def similarity(self, descriptor: Appearance) -> Match:
        if not self.anchors:
            return Match(0, 0, 0, 0, 0)
        anchor = max(self._compare(descriptor, item) for item in self.anchors)
        best = max(
            [anchor, *(self._compare(descriptor, item) for _, item in self.gallery)]
        )
        # An automatically accepted view can help pose changes, but cannot create
        # a chain of drifting templates that no longer resembles a user anchor.
        score = min(best[0], anchor[0] + 0.08)
        negative = max(
            (self._compare(descriptor, item)[0] for item in self.negatives), default=0.0
        )
        return Match(
            score,
            anchor[0],
            best[1],
            best[2],
            anchor[1],
            negative,
            anchor[0] - negative if self.negatives else 1.0,
        )

    def ball_similarity(self, descriptor: Appearance) -> BallMatch:
        """Rank ball-like masks for the caller's velocity/acceleration checks.

        Blurred heads and balls can share these features. In particular,
        ``plausible=True`` must not authorize a global recovery or an update.
        The original ``decide`` identity thresholds remain unchanged.
        """
        anchors = [
            item
            for item in self.anchors
            if item.ball_vector is not None and item.shape is not None
        ]
        if (
            self.target != "ball"
            or descriptor.ball_vector is None
            or descriptor.shape is None
            or not anchors
        ):
            return BallMatch(0, 0, 0, 0, 0, 0, False)
        positive = max(
            float(descriptor.ball_vector @ item.ball_vector) for item in anchors
        )
        negative = max(
            (
                float(descriptor.ball_vector @ item.ball_vector)
                for item in self.negatives
                if item.ball_vector is not None
            ),
            default=0.0,
        )
        shape_score = max(
            1 - float(np.abs(np.asarray(descriptor.shape) - item.shape).mean())
            for item in anchors
        )
        color = max(self._compare(descriptor, item)[2] for item in anchors)
        margin = positive - negative if negative else 1.0
        plausible = positive >= 0.76 and _ball_shape_valid(descriptor.shape)
        # Known, nearly identical negative appearances add a veto; they never
        # lower the semantic or geometric floor for an unrelated candidate.
        plausible &= not (negative >= 0.94 and margin < -0.03)
        original = self.similarity(descriptor)
        plausible &= not (
            original.negative_score >= 0.90 and original.negative_margin < -0.06
        )
        return BallMatch(
            positive,
            negative,
            shape_score,
            color,
            0.80 * positive + 0.15 * shape_score + 0.05 * color,
            margin,
            bool(plausible),
        )

    def decide(
        self,
        candidates: list[tuple[Appearance, float]],
        recovering: bool = True,
    ) -> Selection:
        if not self.anchors or not candidates:
            return Selection(None, False, 0, 0, "no_anchor_or_candidate")
        usable = [
            (self.similarity(item), i, quality)
            for i, (item, quality) in enumerate(candidates)
            # SAM's predicted IoU is not calibrated identity evidence. Correct
            # full-person masks can score below part masks in multimask output.
            if math.isfinite(quality)
        ]
        if not usable:
            return Selection(None, False, 0, 0, "poor_segmentation")
        if self.target == "ball":
            usable = [
                item
                for item in usable
                if candidates[item[1]][0].shape is None
                or _ball_shape_valid(candidates[item[1]][0].shape)
            ]
            if not usable:
                return Selection(None, False, 0, 0, "non_ball_shape")
        usable = [item for item in usable if math.isfinite(item[0].score)]
        if not usable:
            return Selection(None, False, 0, 0, "invalid_appearance")
        ranked = sorted(
            usable,
            key=lambda value: value[0].score,
            reverse=True,
        )
        # A nearly identical, clearly closer known negative must not hide a
        # valid second candidate. Keep uncertain negatives and low positive
        # scores in the competition: a threshold crossing is not evidence that
        # a similar second object disappeared.
        eligible = [
            item
            for item in ranked
            if not (
                item[0].negative_score >= 0.98
                and item[0].negative_margin <= -0.05
            )
        ]
        if not eligible:
            margin = ranked[0][0].score - ranked[1][0].score if len(ranked) > 1 else 1.0
            return Selection(
                None, False, ranked[0][0].score, margin, "matches_known_distractor"
            )
        ranked = eligible
        match, index, _ = ranked[0]
        margin = match.score - ranked[1][0].score if len(ranked) > 1 else 1.0
        # Neither temporal stability nor an unrelated negative can upgrade a
        # semantic match into identity: the same wrong teammate can be stable.
        threshold = 0.94 if recovering else 0.86
        if (
            match.score < threshold
            or match.anchor_score < 0.86
            or match.anchor_part_score < (0.80 if recovering else 0.68)
        ):
            return Selection(None, False, match.score, margin, "appearance_mismatch")
        if match.negative_margin < 0.04:
            return Selection(
                None, False, match.score, margin, "matches_known_distractor"
            )
        if len(ranked) > 1 and margin < (0.045 if recovering else 0.025):
            return Selection(None, False, match.score, margin, "ambiguous_lookalikes")
        strong = (
            match.score >= threshold
            and match.anchor_score >= 0.86
            and match.anchor_part_score >= (0.80 if recovering else 0.74)
            and margin >= 0.08
        )
        return Selection(
            index,
            strong,
            match.score,
            margin,
            "strong_match" if strong else "needs_confirmation",
        )

    def update(
        self, descriptor: Appearance, frame_idx: int, verified: bool = True
    ) -> bool:
        if not verified or not self.anchors or not self.max_templates:
            return False
        if (
            self.target == "ball"
            and descriptor.shape is not None
            and not _ball_shape_valid(descriptor.shape)
        ):
            return False
        match = self.similarity(descriptor)
        if (
            match.score < 0.86
            or match.anchor_score < 0.86
            or match.anchor_part_score < 0.75
            or match.negative_margin < 0.06
        ):
            return False
        existing = [*self.anchors, *(item for _, item in self.gallery)]
        if max(self._compare(descriptor, item)[0] for item in existing) > 0.97:
            return False
        if self.gallery and abs(frame_idx - self.gallery[-1][0]) < 5:
            return False
        self.gallery.append((int(frame_idx), _protected(descriptor)))
        if len(self.gallery) > self.max_templates:
            # Keep the most diverse automatic views, never replacing an anchor.
            redundant = max(
                range(len(self.gallery)),
                key=lambda i: max(
                    self._compare(self.gallery[i][1], other)[0]
                    for j, (_, other) in enumerate(self.gallery)
                    if i != j
                ),
            )
            self.gallery.pop(redundant)
        return True

    def queries(self) -> np.ndarray:
        descriptors = [*self.anchors, *(item for _, item in self.gallery)]
        if not descriptors:
            return np.empty((0, 0), dtype=np.float32)
        return _unit(np.concatenate([item.search_tokens for item in descriptors]))

    def reference_size(self) -> tuple[float, float]:
        if not self.anchors:
            return 16.0, 16.0
        boxes = np.asarray([item.bbox for item in self.anchors])
        return tuple(np.median(boxes[:, 2:] - boxes[:, :2], axis=0).tolist())

    def stats(self) -> dict:
        return {
            "target": self.target,
            "anchors": len(self.anchors),
            "templates": len(self.gallery),
            "negatives": len(self.negatives),
        }


class DinoExtractor:
    """Frozen official local backbone. Full-frame recall and resized mask crops.

    RGB images are uint8 HWC. The cache holds only the current frame's feature
    maps, independent of video duration. Small targets receive complete tiled
    search plus foreground-colour proposals; no part of the image waits for TTL.
    """

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        device: str = "cpu",
        image_size: int = 448,
        model_name: str | None = None,
    ):
        import torch

        repo, checkpoint = Path(repo).resolve(), Path(checkpoint).resolve()
        if not (repo / "dinov3" / "hub" / "backbones.py").is_file():
            raise FileNotFoundError(f"DINOv3 repository not found: {repo}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {checkpoint}")
        if image_size < 64 or image_size % 16:
            raise ValueError(
                "DINOv3 image_size must be a multiple of 16 and at least 64"
            )
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        backbones = importlib.import_module("dinov3.hub.backbones")
        if model_name is None:
            model_name = next(
                (
                    f"dinov3_{name}"
                    for name in (
                        "vith16plus",
                        "vitl16plus",
                        "vits16plus",
                        "vitb16",
                        "vits16",
                        "vitl16",
                    )
                    if name in checkpoint.name
                ),
                None,
            )
        if model_name is None or not hasattr(backbones, model_name):
            raise ValueError("Cannot infer the DINOv3 architecture; supply model_name")
        # The hubconf also imports unrelated evaluation packages. Importing the
        # official backbone directly needs no training dependencies or network.
        model = getattr(backbones, model_name)(pretrained=False)
        model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True),
            strict=True,
        )
        self.model = model.eval().requires_grad_(False).to(device)
        self.torch, self.device, self.image_size = torch, device, image_size
        self.model_name = model_name
        self._image: np.ndarray | None = None
        self._maps: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _encode(self, image: np.ndarray, size: int) -> np.ndarray:
        h, w = image.shape[:2]
        shape = tuple(
            max(16, round(value * size / max(h, w) / 16) * 16) for value in (w, h)
        )
        resized = cv2.resize(image, shape, interpolation=cv2.INTER_CUBIC)
        tensor = (
            self.torch.from_numpy(np.ascontiguousarray(resized))
            .permute(2, 0, 1)
            .float()[None]
            .to(self.device)
            / 255
        )
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[None, :, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[None, :, None, None]
        with self.torch.inference_mode():
            output = self.model.forward_features((tensor - mean) / std)
        patches = output["x_norm_patchtokens"][0].float().cpu().numpy()
        return _unit(patches.reshape(shape[1] // 16, shape[0] // 16, -1))

    def _dense(self, image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        if image is not self._image:
            self._image, self._maps = image, {}
        if box not in self._maps:
            x1, y1, x2, y2 = box
            self._maps[box] = self._encode(image[y1:y2, x1:x2], self.image_size)
        return self._maps[box]

    @staticmethod
    def _crop_box(
        bbox: tuple, shape: tuple, factor: float = 1.3
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = max(8, (x2 - x1) * factor), max(8, (y2 - y1) * factor)
        return (
            max(0, int(cx - w / 2)),
            max(0, int(cy - h / 2)),
            min(shape[1], math.ceil(cx + w / 2)),
            min(shape[0], math.ceil(cy + h / 2)),
        )

    @staticmethod
    def _prototypes(
        features: np.ndarray, weights: np.ndarray, limit: int = 6
    ) -> np.ndarray:
        tokens = features[weights > max(0.10, float(weights.max()) * 0.40)]
        if not len(tokens):
            tokens = features.reshape(-1, features.shape[-1])[
                np.asarray([weights.argmax()])
            ]
        selected = [_unit(tokens.mean(axis=0))]
        for _ in range(min(limit - 1, len(tokens))):
            distances = tokens @ np.asarray(selected).T
            index = int(np.argmin(distances.max(axis=1)))
            if distances[index].max() > 0.94:
                break
            selected.append(tokens[index])
        return np.asarray(selected, dtype=np.float32)

    def _ball_features(self, rgb: np.ndarray, binary: np.ndarray, bbox: tuple):
        """Remove background and average four orientations without a new model."""
        x0, y0, x1, y1 = bbox
        width, height = x1 - x0, y1 - y0
        side = math.ceil(max(width, height) * 1.45)
        image = np.full((side, side, 3), 127, dtype=np.uint8)
        mask = np.zeros((side, side), dtype=np.uint8)
        x, y = (side - width) // 2, (side - height) // 2
        foreground = binary[y0:y1, x0:x1]
        image[y : y + height, x : x + width][foreground] = rgb[y0:y1, x0:x1][foreground]
        mask[y : y + height, x : x + width] = foreground
        views = []
        for rotation in range(4):
            features = self._encode(np.rot90(image, rotation), 224)
            weights = cv2.resize(
                np.rot90(mask, rotation).astype(np.float32),
                (features.shape[1], features.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
            views.append(_unit((features * weights[..., None]).sum(axis=(0, 1))))
        contours, _ = cv2.findContours(
            foreground.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.contourArea(cv2.convexHull(contour))
        shape = (
            min(width, height) / max(width, height),
            float(foreground.sum()) / (width * height),
            min(1.0, 4 * math.pi * area / max(1.0, perimeter**2)),
            min(1.0, area / max(1.0, hull)),
        )
        return _unit(np.asarray(views).mean(axis=0)), shape

    def describe(
        self, rgb: np.ndarray, mask: np.ndarray, target: str | None = None
    ) -> Appearance | None:
        binary = np.squeeze(np.asarray(mask)) > 0
        if binary.shape != rgb.shape[:2]:
            raise ValueError("The identity mask must match the RGB frame dimensions")
        y, x = np.nonzero(binary)
        if len(x) < 4:
            return None
        bbox = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        x1, y1, x2, y2 = self._crop_box(bbox, rgb.shape)
        # Even a 5-pixel ball is independently resized before feature extraction.
        features = self._encode(rgb[y1:y2, x1:x2], 224)
        weights = cv2.resize(
            binary[y1:y2, x1:x2].astype(np.float32),
            (features.shape[1], features.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        vector = _unit((features * weights[..., None]).sum(axis=(0, 1)))
        parts = []
        for rows in np.array_split(np.arange(len(features)), 3):
            parts.append(
                _unit((features[rows] * weights[rows, :, None]).sum(axis=(0, 1)))
                if weights[rows].sum() > 0.01
                else vector
            )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            binary.astype(np.uint8),
            [8, 4, 2],
            [0, 180, 0, 256, 0, 256],
        ).reshape(-1)
        histogram /= max(1, histogram.sum())
        # Contextual tokens have the scale used during dense scene retrieval;
        # object-crop tokens alone do not compare reliably with tiny scene patches.
        side = max(96, (bbox[2] - bbox[0]) * 6, (bbox[3] - bbox[1]) * 6)
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        context = self._crop_box(
            (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), rgb.shape, 1
        )
        dense = self._dense(rgb, context)
        xa, ya, xb, yb = context
        context_weights = cv2.resize(
            binary[ya:yb, xa:xb].astype(np.float32),
            (dense.shape[1], dense.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        queries = np.concatenate(
            (
                self._prototypes(dense, context_weights),
                self._prototypes(features, weights, 3),
            )
        )
        ball, shape = (
            self._ball_features(rgb, binary, bbox) if target == "ball" else (None, None)
        )
        return Appearance(
            vector, np.asarray(parts), histogram, queries, bbox, len(x), ball, shape
        )

    def _color_points(
        self, rgb: np.ndarray, memory: IdentityMemory
    ) -> list[tuple[float, float, float]]:
        # Colour only recalls proposals. It can never accept an identity.
        histogram = np.maximum.reduce([item.color for item in memory.anchors])
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.int32)
        bins = (
            (hsv[..., 0] * 8 // 180) * 8 + (hsv[..., 1] // 64) * 2 + hsv[..., 2] // 128
        )
        probability = histogram[bins]
        binary = (probability >= max(float(histogram.max()) * 0.20, 0.03)).astype(
            np.uint8
        )
        count, _, stats, centers = cv2.connectedComponentsWithStats(binary, 8)
        area = float(np.median([item.area for item in memory.anchors]))
        points = []
        for i in range(1, count):
            x, y, w, h, pixels = stats[i]
            if (
                max(2, area * 0.12) <= pixels <= area * 6
                and 0.35 <= w / max(h, 1) <= 2.8
            ):
                quality = (
                    float(probability[y : y + h, x : x + w].mean())
                    * min(w, h)
                    / max(w, h)
                )
                points.append((*centers[i], quality))
        return sorted(points, key=lambda point: point[2], reverse=True)

    def search(
        self,
        rgb: np.ndarray,
        memory: IdentityMemory,
        limit: int = 6,
        previous_bbox: tuple | None = None,
        tile_offset: int = 0,
    ) -> list[Proposal]:
        # tile_offset is retained for API compatibility; coverage is never rotated.
        del tile_offset
        queries = memory.queries()
        if not len(queries) or limit <= 0:
            return []
        h, w = rgb.shape[:2]
        bw, bh = memory.reference_size()

        def proposal(x, y, score, source="dense"):
            return Proposal(
                (float(x), float(y)),
                (
                    max(0, float(x - bw * 0.65)),
                    max(0, float(y - bh * 0.65)),
                    min(w - 1, float(x + bw * 0.65)),
                    min(h - 1, float(y + bh * 0.65)),
                ),
                float(score),
                source,
            )

        def separated(candidate, others):
            return all(
                ((candidate.point[0] - other.point[0]) / max(8, bw * 0.75)) ** 2
                + ((candidate.point[1] - other.point[1]) / max(8, bh * 0.75)) ** 2
                >= 1
                for other in others
            )

        def heatmap(box):
            dense = self._dense(rgb, box)
            similarity = dense @ queries.T
            top = min(3, len(queries))
            return (
                np.partition(similarity, -top, axis=-1)[..., -top:]
                .mean(axis=-1)
                .astype(np.float32)
            )

        def location(box, heat, row, col):
            return (
                box[0] + (col + 0.5) / heat.shape[1] * (box[2] - box[0]),
                box[1] + (row + 0.5) / heat.shape[0] * (box[3] - box[1]),
            )

        boxes = [(0, 0, w, h)]
        if previous_bbox is not None:
            boxes.append(self._crop_box(previous_bbox, rgb.shape, 5))
        small = memory.target == "ball" or min(bw, bh) < 32 or bw * bh < w * h * 0.003
        if small:
            side = min(max(w, h), max(256, math.ceil(max(w, h) / 3 * 1.2)))
            xs = np.linspace(
                0, max(0, w - side), min(3, math.ceil(w / side)), dtype=int
            )
            ys = np.linspace(
                0, max(0, h - side), min(3, math.ceil(h / side)), dtype=int
            )
            boxes.extend(
                (int(x), int(y), min(w, int(x) + side), min(h, int(y) + side))
                for y in ys
                for x in xs
            )
        candidates: list[Proposal] = []
        for box in dict.fromkeys(boxes):
            heat = heatmap(box)
            peaks = heat >= cv2.dilate(heat, np.ones((3, 3), np.uint8))
            rows, cols = np.nonzero(peaks)
            order = np.argsort(heat[rows, cols])[::-1][: limit * 3]
            for index in order:
                row, col = rows[index], cols[index]
                candidates.append(
                    proposal(*location(box, heat, row, col), heat[row, col])
                )
        ordered = sorted(candidates, key=lambda item: item.similarity, reverse=True)
        selected: list[Proposal] = []
        for candidate in ordered:
            if separated(candidate, selected):
                selected.append(candidate)
            if len(selected) >= limit:
                break
        if small:
            # A ball can be smaller than a *tiled* 16-pixel token too. Re-encode
            # a tight neighbourhood before giving SAM2 its foreground point.
            seeds = selected[:4]
            seeds += [
                proposal(x, y, score, "color")
                for x, y, score in self._color_points(rgb, memory)[:1]
            ]
            refined = []
            side = max(96, 4 * max(bw, bh))
            for seed in seeds:
                x, y = seed.point
                box = self._crop_box(
                    (x - side / 2, y - side / 2, x + side / 2, y + side / 2),
                    rgb.shape,
                    1,
                )
                heat = heatmap(box)
                row, col = np.unravel_index(int(heat.argmax()), heat.shape)
                refined.append(
                    proposal(*location(box, heat, row, col), heat[row, col], "refined")
                )
            selected = []
            for candidate in sorted(
                [*refined, *ordered], key=lambda item: item.similarity, reverse=True
            ):
                if separated(candidate, selected):
                    selected.append(candidate)
                if len(selected) >= limit:
                    break
        return selected

    def release(self) -> None:
        self._image, self._maps = None, {}
        self.model = None


def save_memories(path: Path, memories: dict[str, IdentityMemory]) -> None:
    """Save auditable per-video templates in one NPZ without pickled objects."""
    arrays, metadata = {}, {"version": 1, "targets": {}}
    for target, memory in memories.items():
        entries = []
        for kind, frame, descriptor in [
            *(("anchor", -1, item) for item in memory.anchors),
            *(("gallery", frame, item) for frame, item in memory.gallery),
            *(("negative", -1, item) for item in memory.negatives),
        ]:
            prefix = f"{target}_{len(entries)}"
            for field in ("vector", "parts", "color", "search_tokens"):
                arrays[f"{prefix}_{field}"] = getattr(descriptor, field)
            if descriptor.ball_vector is not None:
                arrays[f"{prefix}_ball_vector"] = descriptor.ball_vector
            entries.append(
                {
                    "prefix": prefix,
                    "kind": kind,
                    "frame": frame,
                    "bbox": descriptor.bbox,
                    "area": descriptor.area,
                    "shape": descriptor.shape,
                }
            )
        metadata["targets"][target] = {
            "max_templates": memory.max_templates,
            "entries": entries,
        }
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    with Path(path).open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def load_memories(path: Path) -> dict[str, IdentityMemory]:
    memories = {}
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("version") != 1:
            raise ValueError("Unsupported identity memory version")
        for target, data in metadata["targets"].items():
            memory = IdentityMemory(target, data["max_templates"])
            for entry in data["entries"]:
                descriptor = Appearance(
                    *(
                        archive[f"{entry['prefix']}_{field}"]
                        for field in ("vector", "parts", "color", "search_tokens")
                    ),
                    tuple(entry["bbox"]),
                    entry["area"],
                    archive[f"{entry['prefix']}_ball_vector"]
                    if f"{entry['prefix']}_ball_vector" in archive
                    else None,
                    tuple(entry["shape"]) if entry.get("shape") is not None else None,
                )
                if entry["kind"] == "anchor":
                    memory.add_anchor(descriptor)
                elif entry["kind"] == "negative":
                    memory.negatives.append(_protected(descriptor))
                else:
                    memory.gallery.append((entry["frame"], _protected(descriptor)))
            memories[target] = memory
    return memories
