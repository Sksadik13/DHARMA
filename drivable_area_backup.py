"""
DHARMA - Dynamic Hazard-aware Autonomous Routing and Motion Adaptation
SIH26037 : Adaptive Path Planning & Collision Avoidance on Unstructured Indian Roads

PERSON 2 MODULE  ->  ROAD & DRIVABLE AREA DETECTION
===========================================================================
Answers exactly one question:  "WHERE CAN THE VEHICLE POTENTIALLY DRIVE?"
It does NOT produce a collision-free trajectory - that is Person 5's job.

Agreed team interface (do not rename):

    from road.drivable_area import detect_drivable_area
    result = detect_drivable_area(frame)          # frame = ONE OpenCV BGR frame

    result = {
        "drivable_area":  mask,             # uint8 HxW, values 0 / 255
        "left_boundary":  [[x, y], ...],    # ordered bottom (near) -> far
        "right_boundary": [[x, y], ...],    # ordered bottom (near) -> far
        "road_center":    [[x, y], ...],    # ordered bottom (near) -> far
        "lane_available": True / False,
    }

Additive extras (safe for Person 5 to ignore):
    corridor_polygon, backend, confidence, look_ahead_ratio,
    corridor_width_px, curvature, direction, horizon_y

This module NEVER opens a video file and NEVER assumes a filename.

Backends, auto-selected best-first:
  1. "segformer"  pretrained SegFormer-B0 finetuned on Cityscapes (has a real
                  'road' class). Nothing is trained here - weights are
                  downloaded once by huggingface and cached.
  2. "classical"  zero-dependency fallback: robust LAB colour-statistics road
                  region growing seeded from the bonnet area + edge
                  suppression + largest connected component.
                  NOTE: this is NOT Canny+Hough line fitting. No lines are
                  ever fitted to edges; edges are only used to CUT the region.

Windows / CPU friendly.  No training.  No MPS.  No GPU assumed.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "detect_drivable_area",
    "DrivableAreaDetector",
    "RoadConfig",
    "configure",
    "reset_state",
    "get_detector",
    "backend_info",
]

# ==========================================================================
# Configuration
# ==========================================================================

#: SegFormer checkpoints tried in order. All are Cityscapes-finetuned, so
#: class id/label "road" exists. b0 is the smallest variant (~3.7M params).
SEGFORMER_CANDIDATES: Tuple[str, ...] = (
    "nvidia/segformer-b0-finetuned-cityscapes-512-1024",
    "nvidia/segformer-b0-finetuned-cityscapes-640-1280",
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
)

#: Label names treated as drivable. Cityscapes 'road' is the ego road surface.
DRIVABLE_LABELS: Tuple[str, ...] = ("road",)


@dataclass
class RoadConfig:
    """Every tunable in one place. All ratios are fractions of frame size."""

    # --- backend -----------------------------------------------------------
    backend: str = "auto"            # "auto" | "segformer" | "classical"
    device: str = "auto"             # "auto" | "cpu" | "cuda"   (never mps)
    model_id: Optional[str] = None   # override SegFormer checkpoint
    seg_long_side: int = 512         # segmentation input long side (CPU speed)
    infer_every: int = 2             # run heavy segmentation every N frames

    # --- mask post-processing ---------------------------------------------
    work_width: int = 320            # resolution used for geometry (fast)
    horizon_ratio: float = 0.38      # everything above this row is ignored
    mask_threshold: float = 0.5      # probability -> binary
    mask_smoothing: float = 0.55     # temporal weight kept from previous frame

    # --- scanline boundary extraction -------------------------------------
    n_rows: int = 22                 # number of sampled scanlines
    row_bottom_ratio: float = 0.995  # nearest scanline
    row_top_ratio: float = 0.46      # furthest scanline we ever probe
    min_run_ratio: float = 0.035     # min drivable run width to accept a row
    max_center_jump_ratio: float = 0.20  # curvature-following gate

    # --- curve fitting / stability ----------------------------------------
    poly_degree: int = 2             # 2 = parabola, follows road curvature
    coef_smoothing: float = 0.65     # temporal weight kept from previous frame
    max_reuse_frames: int = 12       # hold last good geometry this many frames

    # --- corridor ----------------------------------------------------------
    corridor_inset_ratio: float = 0.055   # pull corridor inside the road edges
    min_width_ratio: float = 0.09         # stop the corridor before it needles
    min_bottom_width_ratio: float = 0.10  # needed for lane_available
    min_valid_rows: int = 4               # needed for lane_available

    # --- sanity gates (stop the module from "finding" a road in nonsense) --
    max_coverage: float = 0.97       # mask filling the whole frame = degenerate
    min_contrast: float = 1.5        # flat/blank image = nothing to segment
    degenerate_hold: int = 3         # bad frames tolerated before blanking

    extra_drivable_labels: Tuple[str, ...] = ()

    def merged(self, **overrides: Any) -> "RoadConfig":
        data = {k: getattr(self, k) for k in self.__dataclass_fields__}
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise TypeError(f"RoadConfig has no option '{key}'")
            data[key] = value
        return RoadConfig(**data)


# ==========================================================================
# Small numeric helpers
# ==========================================================================


def _find_runs(row: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Contiguous True runs in a 1-D boolean row, as (start, end_inclusive)."""
    flags = np.concatenate(([0], row.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(flags))
    starts, ends = edges[0::2], edges[1::2]
    keep = (ends - starts) >= max(1, min_len)
    return [(int(s), int(e) - 1) for s, e in zip(starts[keep], ends[keep])]


def _keep_seed_component(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    """Keep the biggest blob that overlaps `seed`; else the biggest blob."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(mask)
    best_seeded = (0, -1)
    best_any = (0, -1)
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_any[1]:
            best_any = (i, area)
        if np.any((labels == i) & (seed > 0)) and area > best_seeded[1]:
            best_seeded = (i, area)
    chosen = best_seeded[0] if best_seeded[0] else best_any[0]
    return np.where(labels == chosen, 255, 0).astype(np.uint8)


def _fit_curve(
    t: np.ndarray, x: np.ndarray, degree: int, weights: Optional[np.ndarray]
) -> np.ndarray:
    """Weighted polyfit x = f(t) with graceful degradation for few points."""
    n = int(t.size)
    if n == 0:
        return np.array([0.0])
    deg = int(min(degree, max(0, n - 2)))
    if n < 3:
        deg = 0 if n == 1 else 1
    try:
        return np.polyfit(t, x, deg, w=weights)
    except Exception:
        return np.array([float(np.mean(x))])


def _blend_coefs(prev: np.ndarray, cur: np.ndarray, keep: float) -> np.ndarray:
    """EMA two polynomial coefficient vectors of possibly different length."""
    size = max(prev.size, cur.size)
    a = np.zeros(size, np.float64)
    b = np.zeros(size, np.float64)
    a[size - prev.size:] = prev
    b[size - cur.size:] = cur
    return keep * a + (1.0 - keep) * b


# ==========================================================================
# Backend 2 : classical road-region growing (always available, no downloads)
# ==========================================================================


class ClassicalRoadBackend:
    """Seeded colour-statistics road segmentation.

    The bonnet area of a forward-facing camera is almost always road, so we
    sample LAB statistics there (median + MAD = outlier resistant) and grow the
    region to every pixel with similar colour. Chroma (a,b) is weighted above
    lightness (L) so shadows and sun patches do not break the region.
    Canny edges are used ONLY as a cutting stencil - no line fitting.
    """

    name = "classical"
    note = "colour-statistics region growing (no pretrained weights)"

    def __init__(self, cfg: RoadConfig) -> None:
        self.cfg = cfg
        self._channel_scale = np.array([0.55, 1.0, 1.0], np.float32)

    def available(self) -> bool:  # pragma: no cover - trivially true
        return True

    def load(self) -> None:
        return None

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        """Return a float32 probability map in [0, 1] at the input frame size."""
        h, w = bgr.shape[:2]
        sw = int(self.cfg.work_width)
        sh = max(16, int(round(sw * h / float(w))))
        small = cv2.resize(bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        blur = cv2.GaussianBlur(small, (5, 5), 0)
        lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

        seed = np.zeros((sh, sw), np.uint8)
        cv2.fillConvexPoly(
            seed,
            np.array(
                [
                    [int(0.28 * sw), sh - 1],
                    [int(0.72 * sw), sh - 1],
                    [int(0.62 * sw), int(0.84 * sh)],
                    [int(0.38 * sw), int(0.84 * sh)],
                ],
                np.int32,
            ),
            255,
        )

        samples = lab[seed > 0]
        median = np.median(samples, axis=0)
        mad = np.median(np.abs(samples - median), axis=0) * 1.4826
        mad = np.maximum(mad, np.array([7.0, 2.5, 2.5], np.float32))
        deviation = (np.abs(lab - median) / mad) * self._channel_scale
        distance = np.sqrt(np.sum(deviation ** 2, axis=2))
        road = (distance < 3.0).astype(np.uint8) * 255

        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.dilate(cv2.Canny(gray, 60, 170), np.ones((3, 3), np.uint8))
        road[edges > 0] = 0
        road[: int(self.cfg.horizon_ratio * sh), :] = 0

        kernel = np.ones((5, 5), np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel, iterations=2)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel, iterations=1)
        road = _keep_seed_component(road, seed)
        road = cv2.GaussianBlur(road, (9, 9), 0)
        return road.astype(np.float32) / 255.0


# ==========================================================================
# Backend 1 : pretrained SegFormer-B0 (Cityscapes) semantic segmentation
# ==========================================================================


class SegformerRoadBackend:
    """Pretrained semantic segmentation, road class only. Nothing is trained.

    Weights are fetched once by huggingface_hub and cached under
    %USERPROFILE%\\.cache\\huggingface, then reused offline.

    Preprocessing (resize + ImageNet normalisation) is done here with numpy
    instead of a transformers ImageProcessor: it is faster, and it keeps the
    module working across transformers versions.
    """

    name = "segformer"
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, cfg: RoadConfig) -> None:
        self.cfg = cfg
        self.note = "not loaded"
        self.model = None
        self.device = "cpu"
        self.model_id: Optional[str] = None
        self.road_ids: List[int] = []
        self._torch = None

    def available(self) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self.note = f"unavailable ({exc.__class__.__name__}: {exc})"
            return False
        return True

    def _resolve_device(self, torch_mod: Any) -> str:
        want = (self.cfg.device or "auto").lower()
        if want == "cpu":
            return "cpu"
        # "auto" and "cuda" both require a real CUDA build. MPS is never used.
        return "cuda" if torch_mod.cuda.is_available() else "cpu"

    def load(self) -> None:
        import torch
        import transformers

        self._torch = torch
        torch.set_grad_enabled(False)
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:
            pass

        loader = getattr(transformers, "AutoModelForSemanticSegmentation", None)
        if loader is None:  # very old transformers
            loader = transformers.SegformerForSemanticSegmentation
        candidates = (
            (self.cfg.model_id,) if self.cfg.model_id else SEGFORMER_CANDIDATES
        )
        last_error: Optional[Exception] = None
        for candidate in candidates:
            try:
                model = loader.from_pretrained(candidate)
            except Exception as exc:  # offline, bad id, corrupt cache, ...
                last_error = exc
                continue
            self.device = self._resolve_device(torch)
            self.model = model.eval().to(self.device)
            self.model_id = candidate
            self.road_ids = self._find_road_ids(dict(model.config.id2label))
            self.note = f"{candidate.split('/')[-1]} on {self.device}"
            return
        raise RuntimeError(f"no pretrained checkpoint could be loaded: {last_error}")

    def _find_road_ids(self, id2label: Dict[Any, str]) -> List[int]:
        wanted = tuple(DRIVABLE_LABELS) + tuple(self.cfg.extra_drivable_labels)
        found = [
            int(key)
            for key, label in id2label.items()
            if str(label).strip().lower() in wanted
        ]
        return found or [0]      # Cityscapes id 0 is 'road'

    def _tensor(self, bgr: np.ndarray) -> Any:
        """BGR uint8 frame -> normalised NCHW float tensor on the device."""
        h, w = bgr.shape[:2]
        scale = max(64, int(self.cfg.seg_long_side)) / float(max(h, w))
        tw = max(32, int(round(w * scale / 32.0)) * 32)
        th = max(32, int(round(h * scale / 32.0)) * 32)
        small = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalised = (rgb - self.IMAGENET_MEAN) / self.IMAGENET_STD
        chw = np.ascontiguousarray(normalised.transpose(2, 0, 1)[None])
        return self._torch.from_numpy(chw).to(self.device)

    def infer(self, bgr: np.ndarray) -> np.ndarray:
        torch = self._torch
        h, w = bgr.shape[:2]
        with torch.inference_mode():
            logits = self.model(pixel_values=self._tensor(bgr)).logits
            probability = logits.softmax(dim=1)[:, self.road_ids].sum(dim=1, keepdim=True)
            probability = torch.nn.functional.interpolate(
                probability, size=(h, w), mode="bilinear", align_corners=False
            )
            return probability[0, 0].float().cpu().numpy()


# ==========================================================================
# Detector
# ==========================================================================


class DrivableAreaDetector:
    """Stateful engine behind :func:`detect_drivable_area`.

    State is only used for temporal stability (a flickering corridor looks
    broken in a demo). The public call signature stays exactly as agreed:
    one frame in, one dict out.
    """

    def __init__(self, config: Optional[RoadConfig] = None, **overrides: Any) -> None:
        base = config or RoadConfig()
        self.cfg = base.merged(**overrides)
        env_backend = os.environ.get("DHARMA_ROAD_BACKEND")
        if env_backend and not overrides.get("backend"):
            self.cfg = self.cfg.merged(backend=env_backend.strip().lower())
        self._lock = threading.Lock()
        self._backend: Any = None
        self.backend_name = "none"
        self.backend_note = "not initialised"
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        """Forget all temporal state (call when switching video source)."""
        self._frame_index = 0
        self._prob_small: Optional[np.ndarray] = None
        self._left_coef: Optional[np.ndarray] = None
        self._right_coef: Optional[np.ndarray] = None
        self._top_t: float = 1.0
        self._stale = 0
        self._confidence = 0.0
        self._degenerate = 0
        self._coverage = 0.0

    def _clear_curves(self) -> None:
        self._left_coef = None
        self._right_coef = None
        self._top_t = 1.0
        self._stale = 0
        self._confidence = 0.0

    def _ensure_backend(self) -> None:
        if self._backend is not None:
            return
        wanted = (self.cfg.backend or "auto").lower()
        if wanted in ("auto", "segformer"):
            candidate = SegformerRoadBackend(self.cfg)
            if candidate.available():
                try:
                    candidate.load()
                    self._backend = candidate
                    self.backend_name = candidate.name
                    self.backend_note = candidate.note
                    return
                except Exception as exc:
                    if wanted == "segformer":
                        raise
                    self.backend_note = f"segformer failed: {exc}"
            elif wanted == "segformer":
                raise RuntimeError(
                    "backend='segformer' requested but torch/transformers are "
                    f"not importable -> {candidate.note}"
                )
            else:
                self.backend_note = candidate.note
        fallback = ClassicalRoadBackend(self.cfg)
        fallback.load()
        self._backend = fallback
        self.backend_name = fallback.name
        self.backend_note = (
            fallback.note if wanted == "classical"
            else f"{fallback.note} [fallback: {self.backend_note}]"
        )

    # -- stage 1 : drivable probability map --------------------------------
    def _probability(self, frame: np.ndarray) -> np.ndarray:
        """Run (or reuse) the backend and return the smoothed work-res map."""
        h, w = frame.shape[:2]
        sw = int(self.cfg.work_width)
        sh = max(16, int(round(sw * h / float(w))))
        due = (
            self._prob_small is None
            or self._prob_small.shape != (sh, sw)
            or self._frame_index % max(1, int(self.cfg.infer_every)) == 0
        )
        if due:
            raw = self._backend.infer(frame)
            small = cv2.resize(raw, (sw, sh), interpolation=cv2.INTER_AREA)
            small = np.clip(small.astype(np.float32), 0.0, 1.0)
            small[: int(self.cfg.horizon_ratio * sh), :] = 0.0
            if self._prob_small is None or self._prob_small.shape != small.shape:
                self._prob_small = small
            else:
                keep = float(np.clip(self.cfg.mask_smoothing, 0.0, 0.95))
                self._prob_small = keep * self._prob_small + (1.0 - keep) * small
        return self._prob_small

    # -- stage 1b : refuse to hallucinate a road ----------------------------
    def _sanity(self, frame: np.ndarray, binary: np.ndarray) -> Optional[str]:
        """Reject degenerate segmentations. Returns a short reason or None.

        A seeded region grower will happily label a blank, saturated or pure
        noise frame as 100% road. Person 5 must never receive that as a usable
        corridor, so those cases are gated here.
        """
        sh, sw = binary.shape[:2]
        y0 = int(self.cfg.horizon_ratio * sh)
        region = binary[y0:]
        if region.size == 0:
            return "frame too small"
        self._coverage = float((region > 0).mean())
        if self._coverage > self.cfg.max_coverage:
            return "mask fills frame"
        gray = cv2.cvtColor(
            cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2GRAY,
        )
        if float(gray[y0:].std()) < self.cfg.min_contrast:
            return "no texture"
        return None

    # -- stage 2 : scanline measurement of left / right road edges ---------
    def _measure(self, binary: np.ndarray) -> List[Tuple[float, float, float, bool]]:
        """Walk scanlines from the bonnet upward, following the road run.

        Returns normalised measurements ``(t, x_left, x_right, clipped)`` in
        0..1 where t = y / height, so the result is resolution independent.
        ``clipped`` marks rows whose run touches the frame border - there the
        true road edge is outside the image, so the sample carries no real
        edge information and is skipped by the curve fit when possible.
        """
        cfg = self.cfg
        sh, sw = binary.shape[:2]
        top = max(cfg.row_top_ratio, cfg.horizon_ratio + 0.02)
        rows = np.linspace(cfg.row_bottom_ratio, top, max(4, cfg.n_rows))
        min_run = max(3, int(cfg.min_run_ratio * sw))
        max_jump = cfg.max_center_jump_ratio * sw

        measured: List[Tuple[float, float, float, bool]] = []
        center: Optional[float] = None
        misses = 0
        for t in rows:
            y = int(np.clip(round(t * sh), 0, sh - 1))
            runs = _find_runs(binary[y] > 0, min_run)
            if not runs:
                misses += 1
                if misses > 2:
                    break
                continue
            if center is None:
                inside = [r for r in runs if r[0] <= sw * 0.5 <= r[1]]
                left, right = max(inside or runs, key=lambda r: r[1] - r[0])
            else:
                inside = [r for r in runs if r[0] <= center <= r[1]]
                if inside:
                    left, right = max(inside, key=lambda r: r[1] - r[0])
                else:
                    left, right = min(
                        runs, key=lambda r: abs(0.5 * (r[0] + r[1]) - center)
                    )
                    if abs(0.5 * (left + right) - center) > max_jump:
                        break
            misses = 0
            center = 0.5 * (left + right)
            clipped = bool(left <= 0 or right >= sw - 1)
            measured.append(
                (float(t), left / float(sw), right / float(sw), clipped)
            )
        return measured

    # -- stage 3 : temporally smoothed curve fit ---------------------------
    def _update_curves(
        self, measured: Sequence[Tuple[float, float, float, bool]]
    ) -> bool:
        cfg = self.cfg
        if len(measured) < 2:
            self._stale += 1
            has_history = self._left_coef is not None
            if has_history and self._stale <= cfg.max_reuse_frames:
                self._confidence *= 0.82
                return True
            return False

        # Rows pinned to the frame border say nothing about the real road
        # edge, so fit on interior rows whenever there are enough of them.
        interior = [m for m in measured if not m[3]]
        source = interior if len(interior) >= 5 else list(measured)
        t = np.array([m[0] for m in source], np.float64)
        xl = np.array([m[1] for m in source], np.float64)
        xr = np.array([m[2] for m in source], np.float64)
        weights = 0.4 + t                      # near rows are more trustworthy
        left = _fit_curve(t, xl, cfg.poly_degree, weights)
        right = _fit_curve(t, xr, cfg.poly_degree, weights)
        top_t = float(min(m[0] for m in measured))

        keep = float(np.clip(cfg.coef_smoothing, 0.0, 0.95))
        if self._left_coef is not None and self._stale <= cfg.max_reuse_frames:
            left = _blend_coefs(self._left_coef, left, keep)
            right = _blend_coefs(self._right_coef, right, keep)
            top_t = keep * self._top_t + (1.0 - keep) * top_t
        self._left_coef, self._right_coef, self._top_t = left, right, top_t
        self._stale = 0
        self._confidence = float(
            np.clip(0.35 + 0.65 * (len(measured) / float(max(4, cfg.n_rows))), 0.0, 1.0)
        )
        return True

    # -- stage 4 : boundaries, centre line and corridor polygon ------------
    def _build_geometry(self, width: int, height: int) -> Dict[str, Any]:
        cfg = self.cfg
        empty = {
            "left": [], "right": [], "center": [], "corridor": None,
            "bottom_width": 0.0, "look_ahead": 0.0, "curvature": 0.0,
            "direction": "unknown", "rows": 0,
        }
        if self._left_coef is None:
            return empty

        top = max(cfg.row_top_ratio, cfg.horizon_ratio + 0.02)
        top_t = float(np.clip(self._top_t, top, cfg.row_bottom_ratio - 0.05))
        t = np.linspace(cfg.row_bottom_ratio, top_t, max(4, cfg.n_rows))
        xl = np.polyval(self._left_coef, t)
        xr = np.polyval(self._right_coef, t)
        low = np.minimum(xl, xr)
        high = np.maximum(xl, xr)
        low = np.clip(low, 0.0, 1.0)
        high = np.clip(high, 0.0, 1.0)

        wide = (high - low) >= cfg.min_width_ratio
        cut = int(np.argmin(wide)) if not bool(np.all(wide)) else len(wide)
        if cut < 2:
            return empty
        t, low, high = t[:cut], low[:cut], high[:cut]

        span = max(1e-6, cfg.row_bottom_ratio - top)
        look_ahead = float(np.clip((cfg.row_bottom_ratio - t[-1]) / span, 0.0, 1.0))
        inset = cfg.corridor_inset_ratio * (high - low)
        c_low = low + inset
        c_high = np.maximum(high - inset, c_low + cfg.min_width_ratio * 0.5)

        ys = np.clip(t * height, 0, height - 1)
        def pts(xn: np.ndarray) -> List[List[int]]:
            xs = np.clip(xn * width, 0, width - 1)
            return [[int(round(x)), int(round(y))] for x, y in zip(xs, ys)]

        left_pts = pts(c_low)
        right_pts = pts(c_high)
        center_pts = pts(0.5 * (c_low + c_high))
        corridor = np.array(left_pts + right_pts[::-1], np.int32)

        centre_norm = 0.5 * (c_low + c_high)
        drift = float(centre_norm[-1] - centre_norm[0])
        direction = "straight"
        if drift < -0.035:
            direction = "left"
        elif drift > 0.035:
            direction = "right"
        curvature = float(self._left_coef[0] + self._right_coef[0]) * 0.5 \
            if self._left_coef.size >= 3 and self._right_coef.size >= 3 else 0.0
        return {
            "left": left_pts, "right": right_pts, "center": center_pts,
            "corridor": corridor,
            "bottom_width": float((high[0] - low[0]) * width),
            "look_ahead": look_ahead, "curvature": curvature,
            "direction": direction, "rows": int(len(t)),
        }

    # -- public entry point ------------------------------------------------
    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """One OpenCV BGR frame in, the agreed result dict out."""
        if frame is None:
            raise ValueError("detect_drivable_area(frame): frame is None")
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR frame, got shape {arr.shape}")
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        height, width = arr.shape[:2]

        with self._lock:
            self._ensure_backend()
            probability = self._probability(arr)
            binary = (probability >= float(self.cfg.mask_threshold)).astype(np.uint8) * 255
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            reason = self._sanity(arr, binary)
            self._degenerate = self._degenerate + 1 if reason else 0
            measured = [] if reason else self._measure(binary)
            if not measured and not reason:
                reason = "no road found"
            usable = self._update_curves(measured)
            if self._degenerate > int(self.cfg.degenerate_hold):
                self._clear_curves()
                usable = False
            geometry = self._build_geometry(width, height) if usable else {
                "left": [], "right": [], "center": [], "corridor": None,
                "bottom_width": 0.0, "look_ahead": 0.0, "curvature": 0.0,
                "direction": "unknown", "rows": 0,
            }
            self._frame_index += 1
            confidence = float(self._confidence)
            coverage = float(self._coverage)
            backend_name, backend_note = self.backend_name, self.backend_note

        mask = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
        wide_enough = geometry["bottom_width"] >= self.cfg.min_bottom_width_ratio * width
        lane_available = bool(
            reason is None
            and geometry["corridor"] is not None
            and geometry["rows"] >= 2
            and wide_enough
            and (len(measured) >= self.cfg.min_valid_rows or confidence >= 0.35)
        )
        return {
            # ---- agreed contract ----------------------------------------
            "drivable_area": mask,
            "left_boundary": geometry["left"],
            "right_boundary": geometry["right"],
            "road_center": geometry["center"],
            "lane_available": lane_available,
            # ---- additive extras ----------------------------------------
            "corridor_polygon": geometry["corridor"],
            "backend": backend_name,
            "backend_note": backend_note,
            "confidence": round(confidence, 3),
            "reject_reason": reason or "",
            "coverage": round(coverage, 3),
            "look_ahead_ratio": round(float(geometry["look_ahead"]), 3),
            "corridor_width_px": round(float(geometry["bottom_width"]), 1),
            "curvature": round(float(geometry["curvature"]), 4),
            "direction": geometry["direction"],
            "horizon_y": int(self.cfg.horizon_ratio * height),
            "frame_index": int(self._frame_index),
        }

    __call__ = process





# ==========================================================================
# Module-level singleton so the agreed one-argument API stays intact
# ==========================================================================

_DETECTOR: Optional[DrivableAreaDetector] = None
_SINGLETON_LOCK = threading.Lock()


def get_detector(**overrides: Any) -> DrivableAreaDetector:
    """Return the shared detector, creating it on first use."""
    global _DETECTOR
    with _SINGLETON_LOCK:
        if _DETECTOR is None:
            _DETECTOR = DrivableAreaDetector(**overrides)
        return _DETECTOR


def configure(**overrides: Any) -> DrivableAreaDetector:
    """Rebuild the shared detector with new options (call before the loop).

    Example::

        from road import drivable_area
        drivable_area.configure(backend="classical", work_width=384)
    """
    global _DETECTOR
    with _SINGLETON_LOCK:
        _DETECTOR = DrivableAreaDetector(**overrides)
        return _DETECTOR


def reset_state() -> None:
    """Clear temporal smoothing - use when switching to another video."""
    with _SINGLETON_LOCK:
        if _DETECTOR is not None:
            _DETECTOR.reset()


def backend_info() -> Dict[str, str]:
    """Which backend is actually running (handy for the demo HUD)."""
    detector = get_detector()
    detector._ensure_backend()
    return {"backend": detector.backend_name, "note": detector.backend_note}


def detect_drivable_area(frame: np.ndarray) -> Dict[str, Any]:
    """AGREED TEAM INTERFACE - Person 2's entry point.

    Parameters
    ----------
    frame : np.ndarray
        One OpenCV BGR frame (HxWx3, uint8), supplied by the caller's video
        loop. This function never opens a file and never assumes a filename.

    Returns
    -------
    dict
        ``drivable_area`` (uint8 mask, 0/255), ``left_boundary``,
        ``right_boundary``, ``road_center`` (lists of ``[x, y]`` ordered near
        -> far) and ``lane_available`` (bool), plus additive extras.
    """
    return get_detector().process(frame)
