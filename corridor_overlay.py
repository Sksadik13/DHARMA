"""
DHARMA - Person 2 : virtual drivable corridor renderer.

Pure visualisation. It consumes the dict returned by
``road.drivable_area.detect_drivable_area(frame)`` and paints:

  * a semi-transparent pink / magenta / purple perspective corridor whose
    opacity fades with distance, so the road stays clearly visible
  * left and right drivable boundaries
  * the estimated centre path (dashed + forward chevrons)
  * a "VIRTUAL DRIVABLE CORRIDOR" label and a small status HUD

Naming rule for the demo: this is the ESTIMATED DRIVABLE CORRIDOR.
It is NOT a guaranteed safe path - Person 4 (risk) and Person 5 (planning)
own that claim.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["draw_corridor", "make_mask_view", "stack_side_by_side"]

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- palette (BGR) --------------------------------------------------------
COLOR_NEAR = (235, 45, 255)        # bright magenta, close to the vehicle
COLOR_FAR = (165, 25, 130)         # deep purple, far away
COLOR_BOUNDARY = (255, 120, 255)
COLOR_BOUNDARY_DARK = (70, 0, 70)
COLOR_CENTER = (255, 240, 255)
COLOR_RUNG = (255, 175, 255)
COLOR_PANEL = (46, 8, 42)
COLOR_TEXT = (255, 215, 255)
COLOR_MUTED = (190, 150, 190)
COLOR_WARN = (0, 175, 255)
COLOR_OK = (130, 255, 175)
COLOR_MASK_EDGE = (255, 255, 130)

ALPHA_NEAR = 0.46
ALPHA_FAR = 0.14

def _lerp(a: Sequence[float], b: Sequence[float], f: float) -> Tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * f)) for x, y in zip(a, b))  # type: ignore


def _as_points(raw: Any) -> List[Tuple[int, int]]:
    if raw is None:
        return []
    return [(int(p[0]), int(p[1])) for p in raw]


def _scale(frame: np.ndarray) -> float:
    """UI scale factor so overlays look identical at 480p and 1080p."""
    return float(np.clip(frame.shape[1] / 1280.0, 0.42, 1.6))


def _blend_region(
    frame: np.ndarray, overlay: np.ndarray, alpha: np.ndarray, box: Tuple[int, int, int, int]
) -> None:
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    base = frame[y0:y1, x0:x1].astype(np.float32)
    top = overlay[y0:y1, x0:x1].astype(np.float32)
    a = alpha[y0:y1, x0:x1][..., None]
    frame[y0:y1, x0:x1] = np.clip(base * (1.0 - a) + top * a, 0, 255).astype(np.uint8)


def _panel(
    frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.62
) -> None:
    """Translucent rounded-ish backdrop so text stays readable over video."""
    height, width = frame.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    tint = np.full_like(roi, COLOR_PANEL, dtype=np.uint8)
    cv2.addWeighted(tint, alpha, roi, 1.0 - alpha, 0, dst=roi)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), COLOR_BOUNDARY_DARK, 1, cv2.LINE_AA)


def _text(
    frame: np.ndarray, msg: str, org: Tuple[int, int], size: float,
    color: Tuple[int, int, int] = COLOR_TEXT, weight: int = 1,
) -> None:
    cv2.putText(frame, msg, org, FONT, size, (0, 0, 0), weight + 2, cv2.LINE_AA)
    cv2.putText(frame, msg, org, FONT, size, color, weight, cv2.LINE_AA)


def _fill_corridor(
    frame: np.ndarray,
    left: List[Tuple[int, int]],
    right: List[Tuple[int, int]],
    strength: float,
) -> None:
    """Depth-graded translucent fill, drawn band by band for perspective."""
    n = min(len(left), len(right))
    if n < 2:
        return
    height, width = frame.shape[:2]
    overlay = np.zeros_like(frame)
    alpha = np.zeros((height, width), np.float32)
    ui = _scale(frame)

    for i in range(n - 1):
        f = i / float(n - 1)                       # 0 = near, 1 = far
        quad = np.array([left[i], right[i], right[i + 1], left[i + 1]], np.int32)
        cv2.fillConvexPoly(overlay, quad, _lerp(COLOR_NEAR, COLOR_FAR, f), cv2.LINE_AA)
        band = (ALPHA_NEAR + (ALPHA_FAR - ALPHA_NEAR) * f) * strength
        cv2.fillConvexPoly(alpha, quad, float(band), cv2.LINE_AA)

    for i in range(0, n, 2):                       # perspective "rungs"
        f = i / float(n - 1)
        rung = (0.36 + (0.12 - 0.36) * f) * strength
        thickness = max(1, int(round(2.4 * ui * (1.0 - 0.5 * f))))
        cv2.line(overlay, left[i], right[i], COLOR_RUNG, thickness, cv2.LINE_AA)
        cv2.line(alpha, left[i], right[i], float(rung), thickness, cv2.LINE_AA)

    poly = np.array(left + right[::-1], np.int32)
    x0, y0, w, h = cv2.boundingRect(poly)
    pad = 3
    box = (max(0, x0 - pad), max(0, y0 - pad),
           min(width, x0 + w + pad), min(height, y0 + h + pad))
    _blend_region(frame, overlay, np.clip(alpha, 0.0, 1.0), box)


def _draw_boundaries(
    frame: np.ndarray,
    left: List[Tuple[int, int]],
    right: List[Tuple[int, int]],
    strong: bool,
) -> None:
    ui = _scale(frame)
    thickness = max(2, int(round(3 * ui)))
    color = COLOR_BOUNDARY if strong else COLOR_MUTED
    for pts in (left, right):
        if len(pts) < 2:
            continue
        arr = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [arr], False, COLOR_BOUNDARY_DARK,
                      thickness + max(2, int(2 * ui)), cv2.LINE_AA)
        cv2.polylines(frame, [arr], False, color, thickness, cv2.LINE_AA)
    for pts in (left, right):                      # anchor dots at the bonnet
        if pts:
            cv2.circle(frame, pts[0], max(3, int(4 * ui)), color, -1, cv2.LINE_AA)


def _draw_center_path(frame: np.ndarray, center: List[Tuple[int, int]]) -> None:
    """Dashed centre line + forward chevrons. Estimated centre, not a plan."""
    if len(center) < 2:
        return
    ui = _scale(frame)
    thickness = max(2, int(round(3 * ui)))
    for i in range(len(center) - 1):
        if i % 3 == 2:                            # gap -> dashed look
            continue
        cv2.line(frame, center[i], center[i + 1], COLOR_BOUNDARY_DARK,
                 thickness + max(2, int(2 * ui)), cv2.LINE_AA)
        cv2.line(frame, center[i], center[i + 1], COLOR_CENTER, thickness, cv2.LINE_AA)

    n = len(center)
    for i in range(1, n - 1, 3):
        x, y = center[i]
        f = i / float(n - 1)
        size = max(3, int(round(11 * ui * (1.0 - 0.65 * f))))
        arm = max(1, int(round(2 * ui * (1.0 - 0.5 * f))))
        cv2.line(frame, (x - size, y + size), (x, y), COLOR_CENTER, arm, cv2.LINE_AA)
        cv2.line(frame, (x + size, y + size), (x, y), COLOR_CENTER, arm, cv2.LINE_AA)


def _overlaps(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _draw_label(
    frame: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    reserved: Optional[Tuple[int, int, int, int]] = None,
) -> None:
    """Caption the corridor, keeping clear of the HUD panel."""
    height, width = frame.shape[:2]
    ui = _scale(frame)
    size = 0.52 * ui + 0.14
    (tw, th), _ = cv2.getTextSize(text, FONT, size, 2)
    pad = int(round(9 * ui)) + 4
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    x = int(np.clip(anchor[0] - box_w // 2, 6, max(6, width - box_w - 6)))
    y = int(np.clip(anchor[1] - box_h - int(14 * ui), 6, max(6, height - box_h - 6)))
    if reserved is not None and _overlaps((x, y, x + box_w, y + box_h), reserved):
        below = reserved[3] + int(10 * ui)
        if below + box_h < height - 6:
            y = below
        else:
            x = int(np.clip(reserved[2] + int(10 * ui), 6,
                            max(6, width - box_w - 6)))
    _panel(frame, x, y, box_w, box_h, alpha=0.68)
    _text(frame, text, (x + pad, y + pad + th - 2), size, COLOR_TEXT, 2)
    if not (x <= anchor[0] <= x + box_w and y <= anchor[1] <= y + box_h):
        start = (int(np.clip(anchor[0], x, x + box_w)),
                 y + box_h if anchor[1] > y else y)
        cv2.line(frame, start, anchor, COLOR_BOUNDARY, max(1, int(2 * ui)), cv2.LINE_AA)


def _hud_layout(
    frame: np.ndarray, result: Dict[str, Any], fps: Optional[float]
) -> Dict[str, Any]:
    """Measure the status panel up-front so other overlays can dodge it."""
    ui = _scale(frame)
    size = 0.42 * ui + 0.17
    title_size = 0.46 * ui + 0.18
    title = "DHARMA | P2 ROAD & DRIVABLE AREA"
    available = bool(result.get("lane_available"))
    rows = [
        ("backend", str(result.get("backend", "?")), COLOR_TEXT),
        ("corridor", "AVAILABLE" if available else "NOT RELIABLE",
         COLOR_OK if available else COLOR_WARN),
        ("look-ahead",
         f"{int(round(float(result.get('look_ahead_ratio', 0.0)) * 100))}%", COLOR_TEXT),
        ("width", f"{int(round(float(result.get('corridor_width_px', 0.0))))} px",
         COLOR_TEXT),
        ("curve", str(result.get("direction", "unknown")).upper(), COLOR_TEXT),
        ("confidence", f"{float(result.get('confidence', 0.0)):.2f}", COLOR_TEXT),
    ]
    if fps is not None:
        rows.append(("fps", f"{fps:.1f}", COLOR_TEXT))
    reason = str(result.get("reject_reason", "") or "")
    if reason and not available:
        rows.insert(2, ("reason", reason, COLOR_WARN))

    def width_of(text: str, scale: float, weight: int) -> int:
        return int(cv2.getTextSize(text, FONT, scale, weight)[0][0])

    key_col = max(width_of(k, size, 1) for k, _, _ in rows) + int(14 * ui) + 6
    val_col = max(width_of(v, size, 1) for _, v, _ in rows)
    pad = int(round(11 * ui)) + 4
    inner = max(width_of(title, title_size, 2), key_col + val_col)
    step = int(round(22 * ui)) + 5
    origin = int(round(10 * ui)) + 4
    box_w, box_h = inner + 2 * pad, 2 * pad + step * (len(rows) + 1)
    return {
        "rect": (origin, origin, origin + box_w, origin + box_h),
        "rows": rows, "title": title, "size": size, "title_size": title_size,
        "pad": pad, "step": step, "key_col": key_col, "ui": ui,
    }


def _draw_hud(frame: np.ndarray, layout: Dict[str, Any]) -> None:
    x0, y0, x1, y1 = layout["rect"]
    pad, step, ui = layout["pad"], layout["step"], layout["ui"]
    _panel(frame, x0, y0, x1 - x0, y1 - y0, alpha=0.62)
    y = y0 + pad + step - int(6 * ui)
    _text(frame, layout["title"], (x0 + pad, y), layout["title_size"], COLOR_TEXT, 2)
    for key, value, colour in layout["rows"]:
        y += step
        _text(frame, key, (x0 + pad, y), layout["size"], COLOR_MUTED, 1)
        _text(frame, value, (x0 + pad + layout["key_col"], y), layout["size"], colour, 1)


def draw_corridor(
    frame: np.ndarray,
    result: Dict[str, Any],
    *,
    fps: Optional[float] = None,
    label: str = "VIRTUAL DRIVABLE CORRIDOR",
    show_label: bool = True,
    show_hud: bool = True,
    show_mask_edge: bool = False,
    copy: bool = True,
) -> np.ndarray:
    """Paint the corridor for one frame and return the annotated image.

    ``result`` is exactly what ``detect_drivable_area(frame)`` returned.
    The source frame is left untouched unless ``copy=False``.
    """
    canvas = frame.copy() if copy else frame
    left = _as_points(result.get("left_boundary"))
    right = _as_points(result.get("right_boundary"))
    center = _as_points(result.get("road_center"))
    available = bool(result.get("lane_available"))
    # Measure the HUD first so the caption can dodge it, but draw it last so
    # the translucent corridor fill never washes over the text.
    layout = _hud_layout(canvas, result, fps) if show_hud else None

    if show_mask_edge:
        mask = result.get("drivable_area")
        if isinstance(mask, np.ndarray) and mask.size:
            contours, _ = cv2.findContours(
                (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(canvas, contours, -1, COLOR_MASK_EDGE,
                             max(1, int(_scale(canvas))), cv2.LINE_AA)

    if left and right:
        _fill_corridor(canvas, left, right, strength=1.0 if available else 0.45)
        _draw_boundaries(canvas, left, right, strong=available)
        _draw_center_path(canvas, center)
        if show_label:
            far = center[-1] if center else left[-1]
            text = label if available else "LOW-CONFIDENCE CORRIDOR"
            _draw_label(canvas, text, (int(far[0]), int(far[1])),
                        layout["rect"] if layout else None)
    else:
        ui = _scale(canvas)
        msg = "NO DRIVABLE CORRIDOR DETECTED"
        note = str(result.get("reject_reason", "") or "")
        size = 0.55 * ui + 0.16
        small = 0.40 * ui + 0.15
        (tw, th), _ = cv2.getTextSize(msg, FONT, size, 2)
        (nw, nh), _ = cv2.getTextSize(note, FONT, small, 1) if note else ((0, 0), 0)
        box_w = max(tw, nw) + 28
        box_h = th + 26 + (nh + 10 if note else 0)
        x = (canvas.shape[1] - max(tw, nw)) // 2
        y = int(canvas.shape[0] * 0.55)
        _panel(canvas, x - 14, y - th - 12, box_w, box_h, alpha=0.6)
        _text(canvas, msg, (x, y), size, COLOR_WARN, 2)
        if note:
            _text(canvas, note, (x, y + nh + 8), small, COLOR_MUTED, 1)

    if layout is not None:
        _draw_hud(canvas, layout)
    return canvas


def make_mask_view(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Raw segmentation mask as a viewable BGR image (for tuning / proof)."""
    mask = result.get("drivable_area")
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return np.zeros_like(frame)
    view = cv2.cvtColor((mask > 0).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    view[:, :, 0] = np.minimum(255, view[:, :, 0])
    view[:, :, 1] = (view[:, :, 1] * 0.25).astype(np.uint8)
    ui = _scale(view)
    _text(view, "RAW DRIVABLE MASK", (int(12 * ui) + 4, int(30 * ui) + 6),
          0.45 * ui + 0.16, COLOR_TEXT, 2)
    return view


def stack_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Join two frames horizontally, matching heights."""
    if a.shape[0] != b.shape[0]:
        scale = a.shape[0] / float(b.shape[0])
        b = cv2.resize(b, (max(1, int(round(b.shape[1] * scale))), a.shape[0]))
    return np.hstack([a, b])