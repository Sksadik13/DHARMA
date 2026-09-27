"""
DHARMA
Dynamic Hazard-aware Autonomous Routing and Motion Adaptation

Final integrated demonstration pipeline.

Visible output:
    - Clean DHARMA HUD
    - Object detection information
    - Separate landscape planner graph window
    - Motion direction arrows
    - Collision-risk information
    - Adaptive path
    - Vehicle speed / action
    - Warning state

Backend:
    - Road/drivable-area detection runs but its corridor is NOT displayed
    - Motion prediction runs continuously
    - Collision-risk assessment runs continuously
    - Adaptive path planning runs continuously

This is a prototype image-space system for SIH demonstration.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from perception.detector import detect_and_track
from road.drivable_area import detect_drivable_area
from road.drivable_area import reset_state as reset_road_state
from motion.motion_perception import MotionPredictor
from safety.risk_engine import RiskEngine
from planning.planner import AdaptivePlanner


# ============================================================
# INPUT / OUTPUT
# ============================================================

INPUT_VIDEO = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "road.mp4"
)

OUTPUT_VIDEO = (
    sys.argv[2]
    if len(sys.argv) > 2
    else "dharma_final.mp4"
)


# ============================================================
# WINDOW LAYOUT
# ============================================================

VIDEO_WINDOW_TITLE = "DHARMA — Annotated Road Video"
GRAPH_WINDOW_TITLE = "DHARMA — Planner Graph"
VIDEO_SCREEN_RATIO = 0.70
GRAPH_SCREEN_RATIO = 0.30
PLAYBACK_SPEED = 2.0
PLAYBACK_DELAY_MS = max(1, int(33 / PLAYBACK_SPEED))

# SPACE toggles playback pause/resume.
paused = False

# Matplotlib is considerably heavier than OpenCV. The DHARMA pipeline
# still runs P1-P5 on every frame, but the separate planner graph is
# refreshed every few frames to make the live video display smoother.
# The saved video is still written at the original input FPS.
GRAPH_UPDATE_INTERVAL = 1


def get_screen_size() -> Tuple[int, int]:
    """Return the primary screen size, with a safe fallback."""

    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        screen_width = int(user32.GetSystemMetrics(0))
        screen_height = int(user32.GetSystemMetrics(1))

        if screen_width > 0 and screen_height > 0:
            return screen_width, screen_height
    except Exception:
        pass

    return 1920, 1080


def move_native_window(
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
) -> bool:
    """Move/resize a native Windows window by its exact title."""

    try:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, title)

        if not hwnd:
            return False

        return bool(
            user32.MoveWindow(
                hwnd,
                int(x),
                int(y),
                int(width),
                int(height),
                True,
            )
        )
    except Exception:
        return False


def arrange_demo_windows() -> None:
    """Arrange video on the left 70% and graph on the right 30%."""

    screen_width, screen_height = get_screen_size()

    video_width = int(screen_width * VIDEO_SCREEN_RATIO)
    graph_width = screen_width - video_width

    # Video occupies the larger left area.
    move_native_window(
        VIDEO_WINDOW_TITLE,
        0,
        0,
        video_width,
        screen_height,
    )

    # Planner graph occupies the smaller right area.
    move_native_window(
        GRAPH_WINDOW_TITLE,
        video_width,
        0,
        graph_width,
        screen_height,
    )


# ============================================================
# UI HELPERS
# ============================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.72,
) -> np.ndarray:
    """
    Draw a dark translucent information panel.
    """

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y),
        (x + w, y + h),
        (10, 15, 22),
        -1,
    )

    frame = cv2.addWeighted(
        overlay,
        alpha,
        frame,
        1.0 - alpha,
        0,
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        (100, 110, 120),
        1,
    )

    return frame


def put_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    scale: float = 0.55,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    """
    Draw readable HUD text.
    """

    cv2.putText(
        frame,
        text,
        position,
        FONT,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


# ============================================================
# MOTION DIRECTION
# ============================================================

def normalize_motion_direction(
    prediction: Optional[Dict[str, Any]],
) -> str:
    """
    Convert P3's numeric direction angle into a human-readable
    vehicle-motion direction.

    P3 direction:
        0 degrees   -> right
        90 degrees  -> down
        180 degrees -> left
        270 degrees -> up

    For the DHARMA camera view we classify motion into:
        LEFT
        FORWARD
        RIGHT
        BACKWARD
        STATIONARY
    """

    if not prediction:
        return "UNKNOWN"

    speed = float(
        prediction.get(
            "speed",
            0.0,
        )
        or 0.0
    )

    if speed < 8.0:
        return "STATIONARY"

    direction = prediction.get(
        "direction",
        None,
    )

    if direction is None:
        return "UNKNOWN"

    try:
        angle = float(direction) % 360.0
    except (
        TypeError,
        ValueError,
    ):
        return "UNKNOWN"

    # P3 uses image-space velocity:
    #
    # 0°   = right
    # 90°  = down
    # 180° = left
    # 270° = up
    #
    # Forward relative to the camera is therefore upward.

    if 45.0 <= angle < 135.0:
        return "BACKWARD"

    if 135.0 <= angle < 225.0:
        return "LEFT"

    if 225.0 <= angle < 315.0:
        return "FORWARD"

    return "RIGHT"


# ============================================================
# MOTION ARROW
# ============================================================

def draw_motion_arrow(
    frame: np.ndarray,
    obj: Dict[str, Any],
    prediction: Optional[Dict[str, Any]],
) -> None:
    """
    Draw a compact motion vector over the detected object.

    The vector is based on P3's actual estimated velocity.
    """

    if prediction is None:
        return

    velocity = prediction.get(
        "velocity",
        [0.0, 0.0],
    )

    if not isinstance(
        velocity,
        (list, tuple),
    ):
        return

    if len(velocity) < 2:
        return

    try:
        vx = float(velocity[0])
        vy = float(velocity[1])
    except (
        TypeError,
        ValueError,
    ):
        return

    speed = math.sqrt(
        vx * vx
        + vy * vy
    )

    if speed < 8.0:
        return

    center = obj.get(
        "center",
        None,
    )

    if center is None:
        bbox = obj.get(
            "bbox",
            None,
        )

        if (
            not bbox
            or len(bbox) != 4
        ):
            return

        center = [
            (
                float(bbox[0])
                + float(bbox[2])
            )
            / 2.0,
            (
                float(bbox[1])
                + float(bbox[3])
            )
            / 2.0,
        ]

    try:
        cx = int(center[0])
        cy = int(center[1])
    except (
        TypeError,
        ValueError,
        IndexError,
    ):
        return

    # Keep arrows visually compact.
    arrow_length = int(
        max(
            24.0,
            min(
                65.0,
                speed * 0.35,
            ),
        )
    )

    ux = vx / speed
    uy = vy / speed

    end_x = int(
        cx
        + ux * arrow_length
    )

    end_y = int(
        cy
        + uy * arrow_length
    )

    cv2.arrowedLine(
        frame,
        (cx, cy),
        (end_x, end_y),
        (255, 210, 60),
        2,
        cv2.LINE_AA,
        tipLength=0.28,
    )


# ============================================================
# MOTION FLOW GRAPH
# ============================================================

def draw_motion_flow(
    frame: np.ndarray,
    objects: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> None:
    """
    Draw the live MOTION FLOW graph.

    Shows:
        LEFT
        FORWARD
        RIGHT
        STATIONARY

    Also shows a compact list of active tracked objects.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Prediction lookup
    # --------------------------------------------------------

    prediction_by_id = {
        prediction.get("id"): prediction
        for prediction in predictions
        if prediction.get("id") is not None
    }

    # --------------------------------------------------------
    # Direction counts
    # --------------------------------------------------------

    counts = {
        "LEFT": 0,
        "FORWARD": 0,
        "RIGHT": 0,
        "STATIONARY": 0,
    }

    motion_rows = []

    for obj in objects:

        object_id = obj.get(
            "id",
            None,
        )

        prediction = prediction_by_id.get(
            object_id
        )

        direction = normalize_motion_direction(
            prediction
        )

        if direction in counts:
            counts[direction] += 1

        speed = 0.0

        if prediction is not None:
            try:
                speed = float(
                    prediction.get(
                        "speed",
                        0.0,
                    )
                    or 0.0
                )
            except (
                TypeError,
                ValueError,
            ):
                speed = 0.0

        class_name = str(
            obj.get(
                "class",
                "object",
            )
        ).upper()

        motion_rows.append(
            (
                object_id,
                class_name,
                direction,
                speed,
            )
        )

    # --------------------------------------------------------
    # Panel dimensions
    # --------------------------------------------------------

    panel_w = min(
        340,
        max(
            290,
            int(width * 0.30),
        ),
    )

    panel_h = 250

    panel_x = (
        width
        - panel_w
        - 18
    )

    panel_y = 18

    frame = draw_panel(
        frame,
        panel_x,
        panel_y,
        panel_w,
        panel_h,
        alpha=0.78,
    )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    put_text(
        frame,
        "MOTION FLOW",
        (
            panel_x + 16,
            panel_y + 27,
        ),
        0.68,
        (255, 255, 255),
        2,
    )

    put_text(
        frame,
        "REAL-TIME OBJECT MOVEMENT",
        (
            panel_x + 16,
            panel_y + 47,
        ),
        0.40,
        (170, 180, 190),
        1,
    )

    # --------------------------------------------------------
    # Graph area
    # --------------------------------------------------------

    graph_top = panel_y + 62
    graph_bottom = panel_y + 142

    categories = [
        ("LEFT", counts["LEFT"]),
        ("FORWARD", counts["FORWARD"]),
        ("RIGHT", counts["RIGHT"]),
        ("STOP", counts["STATIONARY"]),
    ]

    graph_left = panel_x + 18
    graph_right = (
        panel_x
        + panel_w
        - 18
    )

    available_width = (
        graph_right
        - graph_left
    )

    column_width = int(
        available_width
        / len(categories)
    )

    max_count = max(
        1,
        max(
            counts.values()
        ),
    )

    for index, (
        label,
        count,
    ) in enumerate(categories):

        cx = (
            graph_left
            + column_width * index
            + column_width // 2
        )

        bar_max_height = 55

        bar_height = int(
            (
                count
                / max_count
            )
            * bar_max_height
        )

        x1 = cx - 18
        x2 = cx + 18

        y1 = (
            graph_bottom
            - bar_height
        )

        y2 = graph_bottom

        if count > 0:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (80, 190, 255),
                -1,
            )
        else:
            cv2.rectangle(
                frame,
                (
                    x1,
                    graph_bottom - 2,
                ),
                (
                    x2,
                    graph_bottom,
                ),
                (70, 80, 90),
                -1,
            )

        put_text(
            frame,
            str(count),
            (
                cx - 7,
                y1 - 7,
            ),
            0.48,
            (255, 255, 255),
            1,
        )

        label_size = 0.36

        put_text(
            frame,
            label,
            (
                cx - 27,
                graph_bottom + 20,
            ),
            label_size,
            (190, 200, 210),
            1,
        )

    # --------------------------------------------------------
    # Object list
    # --------------------------------------------------------

    list_y = panel_y + 166

    put_text(
        frame,
        f"ACTIVE OBJECTS: {len(objects)}",
        (
            panel_x + 16,
            list_y,
        ),
        0.45,
        (255, 255, 255),
        1,
    )

    list_y += 20

    # Show up to four most useful entries.
    for (
        object_id,
        class_name,
        direction,
        speed,
    ) in motion_rows[:4]:

        if object_id is None:
            object_label = "OBJ"
        else:
            object_label = f"#{object_id}"

        arrow = {
            "LEFT": "<",
            "FORWARD": "^",
            "RIGHT": ">",
            "BACKWARD": "v",
            "STATIONARY": "•",
            "UNKNOWN": "?",
        }.get(
            direction,
            "?",
        )

        text = (
            f"{object_label:<5} "
            f"{class_name[:9]:<9} "
            f"{arrow} {direction:<10} "
            f"{speed:>5.1f}"
        )

        put_text(
            frame,
            text,
            (
                panel_x + 16,
                list_y,
            ),
            0.37,
            (215, 220, 225),
            1,
        )

        list_y += 17


# ============================================================
# PREDICTED MOTION PATH
# ============================================================

def draw_prediction_paths(
    frame: np.ndarray,
    objects: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
) -> None:
    """
    Draw subtle future motion trajectories.

    These are intentionally subtle so the final demo does not
    become visually cluttered.
    """

    prediction_by_id = {
        prediction.get("id"): prediction
        for prediction in predictions
        if prediction.get("id") is not None
    }

    object_by_id = {
        obj.get("id"): obj
        for obj in objects
        if obj.get("id") is not None
    }

    for object_id, prediction in prediction_by_id.items():

        obj = object_by_id.get(
            object_id
        )

        if obj is None:
            continue

        predicted_path = prediction.get(
            "predicted_path",
            [],
        )

        if len(predicted_path) < 2:
            continue

        points = []

        for point in predicted_path:

            if (
                not isinstance(
                    point,
                    (list, tuple),
                )
                or len(point) < 2
            ):
                continue

            try:
                points.append(
                    (
                        int(point[0]),
                        int(point[1]),
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

        if len(points) < 2:
            continue

        # Draw a subtle dotted prediction.
        for index in range(
            len(points) - 1
        ):

            if index % 2 != 0:
                continue

            cv2.line(
                frame,
                points[index],
                points[
                    min(
                        index + 1,
                        len(points) - 1,
                    )
                ],
                (255, 150, 80),
                2,
                cv2.LINE_AA,
            )


# ============================================================
# RISK VISUALIZATION
# ============================================================

def risk_color(
    risk: str,
) -> Tuple[int, int, int]:
    """
    Return BGR color for risk level.
    """

    risk = str(
        risk
    ).upper()

    if risk == "HIGH_RISK":
        return (
            70,
            70,
            255,
        )

    if risk == "CAUTION":
        return (
            0,
            190,
            255,
        )

    return (
        80,
        210,
        100,
    )


def draw_risks(
    frame: np.ndarray,
    objects: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
) -> None:
    """
    Draw compact collision-risk information.
    """

    risk_by_id = {
        result.get("object_id"): result
        for result in risks
    }

    for obj in objects:

        object_id = obj.get(
            "id",
            None,
        )

        result = risk_by_id.get(
            object_id
        )

        if result is None:
            continue

        bbox = obj.get(
            "bbox",
            None,
        )

        if (
            not bbox
            or len(bbox) != 4
        ):
            continue

        try:
            x1, y1, x2, y2 = map(
                int,
                bbox,
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        risk = str(
            result.get(
                "risk",
                "SAFE",
            )
        )

        box_color = risk_color(
            risk
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2,
        )

        ttc = result.get(
            "ttc",
            None,
        )

        if ttc is None:
            ttc_text = "--"
        else:
            try:
                ttc_text = (
                    f"{float(ttc):.1f}s"
                )
            except (
                TypeError,
                ValueError,
            ):
                ttc_text = "--"

        class_name = str(
            obj.get(
                "class",
                "object",
            )
        ).upper()

        label = (
            f"#{object_id} "
            f"{class_name} "
            f"{risk}"
        )

        label_y = max(
            20,
            y1 - 8,
        )

        put_text(
            frame,
            label,
            (
                x1,
                label_y,
            ),
            0.42,
            box_color,
            2,
        )

        # TTC beneath object label.
        put_text(
            frame,
            f"TTC {ttc_text}",
            (
                x1,
                min(
                    frame.shape[0] - 8,
                    y2 + 17,
                ),
            ),
            0.38,
            box_color,
            1,
        )


# ============================================================
# FINAL PATH VISUALIZATION
# ============================================================

def draw_planned_path(
    frame: np.ndarray,
    path: Optional[List[List[float]]],
) -> None:
    """
    Draw the final selected trajectory from the planner.
    """

    if not path or len(path) < 2:
        return

    points = []

    for point in path:

        if (
            not isinstance(
                point,
                (list, tuple),
            )
            or len(point) < 2
        ):
            continue

        try:
            points.append(
                (
                    int(point[0]),
                    int(point[1]),
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

    if len(points) < 2:
        return

    for i in range(
        len(points) - 1
    ):

        cv2.line(
            frame,
            points[i],
            points[i + 1],
            (60, 220, 120),
            4,
            cv2.LINE_AA,
        )

    # Direction marker.
    if len(points) >= 2:

        start = points[-2]
        end = points[-1]

        cv2.arrowedLine(
            frame,
            start,
            end,
            (60, 220, 120),
            4,
            cv2.LINE_AA,
            tipLength=0.25,
        )


# ============================================================
# STATUS
# ============================================================

def get_status(
    plan: Dict[str, Any],
) -> str:
    """
    Convert planner action into a clean human-readable status.
    """

    action = str(
        plan.get(
            "action",
            "KEEP_SPEED",
        )
    ).upper()

    if action == "EMERGENCY_BRAKE":
        return "EMERGENCY"

    if action == "BRAKE":
        return "BRAKING"

    if action == "SLOW_DOWN":
        return "SLOWING"

    if action == "CAUTION":
        return "CAUTION"

    if action == "REROUTE":
        return "REROUTING"

    return "NORMAL"



# ============================================================
# SEPARATE PLANNER GRAPH
# ============================================================

GRAPH_WIDTH = 1200
GRAPH_HEIGHT = 700

# Single-window demo layout. The annotated video and planner graph are
# composed into ONE OpenCV window. The saved MP4 remains video-only.
DEMO_WINDOW_TITLE = "DHARMA — Autonomous Driving Dashboard"
DEMO_VIDEO_RATIO = 0.70
DEMO_GRAPH_RATIO = 0.30
DEMO_CANVAS_HEIGHT = 720
DEMO_GRAPH_HEIGHT_RATIO = 0.62
DEMO_STATUS_HEIGHT_RATIO = 0.38


# Demo image-to-planner scale.
# DHARMA's current P2/P5 stack is image-space and does not have a
# calibrated camera model. These constants provide a stable planner
# coordinate view while keeping the conversion in one place.
# Replace them with camera calibration values when available.
PLANNER_HALF_WIDTH_M = 6.0
PLANNER_DEPTH_M = 50.0


class PlannerGraph:
    """
    Reusable landscape planner graph.

    The Matplotlib figure and all artists are created once and updated
    frame-by-frame. The figure is rendered to an image and composed into
    the single DHARMA OpenCV dashboard window.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        planner: AdaptivePlanner,
    ) -> None:
        self.frame_width = max(1, int(frame_width))
        self.planner = planner
        self.frame_height = max(1, int(frame_height))
        self.quit_requested = False

        # The figure is used only as an off-screen renderer.
        # Do not create a second native Matplotlib window.
        plt.ioff()

        self.fig, self.ax = plt.subplots(
            figsize=(12.0, 7.0),
            dpi=100,
        )

        self.fig.subplots_adjust(
            left=0.08,
            right=0.78,
            bottom=0.09,
            top=0.90,
        )

        self.ax.set_xlim(
            -PLANNER_HALF_WIDTH_M,
            PLANNER_HALF_WIDTH_M,
        )
        self.ax.set_ylim(
            0.0,
            PLANNER_DEPTH_M,
        )
        self.ax.set_aspect("auto")

        self.ax.set_xlabel(
            "Lateral position (m)"
        )
        self.ax.set_ylabel(
            "Longitudinal distance (m)"
        )
        self.ax.set_title(
            "DHARMA — Adaptive Planner / Safety View",
            fontsize=15,
            fontweight="bold",
        )

        self.ax.grid(
            True,
            alpha=0.22,
            linewidth=0.8,
        )

        # Fixed artists.
        self.left_boundary, = self.ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=2.0,
            label="Road boundary",
        )
        self.right_boundary, = self.ax.plot(
            [],
            [],
            linestyle="--",
            linewidth=2.0,
        )

        self.selected_path, = self.ax.plot(
            [],
            [],
            linewidth=4.0,
            label="Selected path",
        )

        self.ego_point, = self.ax.plot(
            [],
            [],
            marker="o",
            markersize=10,
            linestyle="None",
            label="Ego vehicle",
        )

        # Candidate path artists are fixed at P5's normal five candidates.
        self.candidate_lines = []
        for _ in range(5):
            line, = self.ax.plot(
                [],
                [],
                linestyle=":",
                linewidth=1.2,
                alpha=0.45,
            )
            self.candidate_lines.append(line)

        # Up to 20 tracked-object prediction lines are reused.
        self.prediction_lines = []
        for _ in range(20):
            line, = self.ax.plot(
                [],
                [],
                linewidth=1.6,
                alpha=0.65,
            )
            self.prediction_lines.append(line)

        self.risk_scatter = self.ax.scatter(
            [],
            [],
            s=90,
            marker="x",
            linewidths=2.5,
        )

        self.object_scatter = self.ax.scatter(
            [],
            [],
            s=35,
            marker="o",
            alpha=0.75,
        )

        # Stable information block: location never changes.
        self.info_text = self.ax.text(
            0.02,
            0.97,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10.5,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.45",
                alpha=0.82,
            ),
        )

        # Permanent legend. Dynamic object labels are NOT added to it.
        legend_handles = [
            Line2D(
                [0],
                [0],
                linestyle="--",
                linewidth=2,
                label="Road boundary",
            ),
            Line2D(
                [0],
                [0],
                linestyle=":",
                linewidth=1.5,
                label="Candidate paths",
            ),
            Line2D(
                [0],
                [0],
                linewidth=4,
                label="Selected path",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="None",
                markersize=9,
                label="Ego vehicle",
            ),
            Line2D(
                [0],
                [0],
                linewidth=1.6,
                label="Predicted trajectories",
            ),
            Line2D(
                [0],
                [0],
                marker="x",
                linestyle="None",
                markersize=9,
                markeredgewidth=2,
                label="Risk marker",
            ),
        ]

        self.ax.legend(
            handles=legend_handles,
            loc="upper left",
            bbox_to_anchor=(1.015, 1.0),
            borderaxespad=0.0,
            framealpha=0.92,
            title="Legend",
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self._on_key,
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _on_key(self, event: Any) -> None:
        if str(event.key).lower() in {"q", "escape"}:
            self.quit_requested = True

    def pixel_to_planner(
        self,
        point: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Convert image pixels to stable planner coordinates.

        x: lateral position relative to image center.
        y: forward/longitudinal position using the image horizon.

        This is a calibrated-style normalized projection for the current
        image-space prototype. True physical metres require camera
        intrinsics/extrinsics or a road-scale calibration.
        """

        x, y = float(point[0]), float(point[1])

        horizon_y = self.frame_height * 0.45
        bottom_y = self.frame_height * 0.92

        x_m = (
            (x - self.frame_width * 0.5)
            / (self.frame_width * 0.5)
            * PLANNER_HALF_WIDTH_M
        )

        denom = max(
            1.0,
            bottom_y - horizon_y,
        )

        y_m = (
            (bottom_y - y)
            / denom
            * PLANNER_DEPTH_M
        )

        return x_m, y_m

    def _convert_path(
        self,
        path: Any,
    ) -> Tuple[List[float], List[float]]:
        xs: List[float] = []
        ys: List[float] = []

        if not path:
            return xs, ys

        for point in path:
            if (
                not isinstance(point, (list, tuple))
                or len(point) < 2
            ):
                continue

            try:
                x_m, y_m = self.pixel_to_planner(
                    (point[0], point[1])
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            xs.append(x_m)
            ys.append(y_m)

        return xs, ys

    def _set_line_from_path(
        self,
        line: Any,
        path: Any,
    ) -> None:
        xs, ys = self._convert_path(path)
        line.set_data(xs, ys)

    def update(
        self,
        road: Dict[str, Any],
        objects: List[Dict[str, Any]],
        predictions: List[Dict[str, Any]],
        risks: List[Dict[str, Any]],
        plan: Dict[str, Any],
        frame_number: int,
    ) -> None:
        """
        Update all graph artists from exactly one pipeline frame.
        """

        # ----------------------------------------------------
        # Road boundaries
        # ----------------------------------------------------

        self._set_line_from_path(
            self.left_boundary,
            road.get("left_boundary", []),
        )

        self._set_line_from_path(
            self.right_boundary,
            road.get("right_boundary", []),
        )

        # ----------------------------------------------------
        # Candidate paths generated by the same P5 planner
        # ----------------------------------------------------

        try:
            candidate_paths = self.planner.generate_candidate_paths(
                self.frame_width,
                self.frame_height,
            )
        except Exception:
            candidate_paths = []

        for index, line in enumerate(
            self.candidate_lines
        ):
            if index < len(candidate_paths):
                self._set_line_from_path(
                    line,
                    candidate_paths[index].get(
                        "points",
                        candidate_paths[index].get(
                            "path",
                            [],
                        ),
                    )
                    if isinstance(
                        candidate_paths[index],
                        dict,
                    )
                    else candidate_paths[index],
                )
                line.set_visible(True)
            else:
                line.set_data([], [])
                line.set_visible(False)

        # ----------------------------------------------------
        # Selected path
        # ----------------------------------------------------

        selected_path = plan.get(
            "path",
            None,
        )

        self._set_line_from_path(
            self.selected_path,
            selected_path,
        )

        # ----------------------------------------------------
        # Ego vehicle
        # ----------------------------------------------------

        ego_pixel = (
            self.frame_width * 0.5,
            self.frame_height * 0.92,
        )

        ego_x, ego_y = self.pixel_to_planner(
            ego_pixel
        )

        self.ego_point.set_data(
            [ego_x],
            [ego_y],
        )

        # ----------------------------------------------------
        # Object positions
        # ----------------------------------------------------

        object_xs: List[float] = []
        object_ys: List[float] = []
        object_labels: List[Tuple[float, float, str]] = []

        object_by_id = {}

        for obj in objects:
            object_id = obj.get("id")
            object_by_id[object_id] = obj

            center = obj.get("center")

            if center is None:
                bbox = obj.get("bbox")
                if (
                    not bbox
                    or len(bbox) != 4
                ):
                    continue

                center = [
                    (
                        float(bbox[0])
                        + float(bbox[2])
                    )
                    / 2.0,
                    (
                        float(bbox[1])
                        + float(bbox[3])
                    )
                    / 2.0,
                ]

            try:
                x_m, y_m = self.pixel_to_planner(
                    (center[0], center[1])
                )
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

            if (
                -PLANNER_HALF_WIDTH_M
                <= x_m
                <= PLANNER_HALF_WIDTH_M
                and 0.0
                <= y_m
                <= PLANNER_DEPTH_M
            ):
                object_xs.append(x_m)
                object_ys.append(y_m)

                object_labels.append(
                    (
                        x_m,
                        y_m,
                        f"#{object_id} "
                        f"{str(obj.get('class', 'object')).upper()}",
                    )
                )

        self.object_scatter.set_offsets(
            np.column_stack(
                (object_xs, object_ys)
            )
            if object_xs
            else np.empty((0, 2))
        )

        # ----------------------------------------------------
        # Predicted trajectories
        # ----------------------------------------------------

        prediction_by_id = {
            prediction.get("id"): prediction
            for prediction in predictions
            if prediction.get("id") is not None
        }

        for index, line in enumerate(
            self.prediction_lines
        ):
            line.set_data([], [])
            line.set_visible(False)

        prediction_index = 0

        for object_id, prediction in prediction_by_id.items():
            if prediction_index >= len(
                self.prediction_lines
            ):
                break

            path = prediction.get(
                "predicted_path",
                [],
            )

            line = self.prediction_lines[
                prediction_index
            ]

            self._set_line_from_path(
                line,
                path,
            )

            line.set_visible(True)
            prediction_index += 1

        # ----------------------------------------------------
        # Risk markers
        # ----------------------------------------------------

        risk_by_id = {
            risk.get("object_id"): risk
            for risk in risks
        }

        risk_xs: List[float] = []
        risk_ys: List[float] = []
        risk_colors: List[str] = []

        for object_id, obj in object_by_id.items():
            result = risk_by_id.get(object_id)

            if result is None:
                continue

            risk_level = str(
                result.get("risk", "SAFE")
            ).upper()

            if risk_level == "SAFE":
                continue

            center = obj.get("center")

            if center is None:
                bbox = obj.get("bbox")
                if (
                    not bbox
                    or len(bbox) != 4
                ):
                    continue

                center = [
                    (
                        float(bbox[0])
                        + float(bbox[2])
                    )
                    / 2.0,
                    (
                        float(bbox[1])
                        + float(bbox[3])
                    )
                    / 2.0,
                ]

            try:
                x_m, y_m = self.pixel_to_planner(
                    (center[0], center[1])
                )
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

            if risk_level == "HIGH_RISK":
                marker_color = "red"
            else:
                marker_color = "orange"

            risk_xs.append(x_m)
            risk_ys.append(y_m)
            risk_colors.append(marker_color)

        self.risk_scatter.set_offsets(
            np.column_stack(
                (risk_xs, risk_ys)
            )
            if risk_xs
            else np.empty((0, 2))
        )

        if risk_colors:
            self.risk_scatter.set_color(
                risk_colors
            )
        else:
            self.risk_scatter.set_color([])

        # ----------------------------------------------------
        # Current planner opinion
        # ----------------------------------------------------

        action = str(
            plan.get(
                "action",
                "KEEP_SPEED",
            )
        ).replace(
            "_",
            " ",
        )

        direction = str(
            plan.get(
                "direction",
                "CENTER",
            )
        ).upper()

        try:
            speed = float(
                plan.get(
                    "speed",
                    0.0,
                )
                or 0.0
            )
        except (
            TypeError,
            ValueError,
        ):
            speed = 0.0

        min_ttc = "--"

        ttc_values = []

        for risk in risks:
            value = risk.get("ttc")
            if value is None:
                continue

            try:
                value = float(value)
            except (
                TypeError,
                ValueError,
            ):
                continue

            if value >= 0:
                ttc_values.append(value)

        if ttc_values:
            min_ttc = f"{min(ttc_values):.1f}s"

        self.info_text.set_text(
            f"FRAME {frame_number:05d}\n"
            f"ACTION: {action}\n"
            f"PATH: {direction}\n"
            f"TARGET SPEED: {speed:.1f}\n"
            f"MIN TTC: {min_ttc}"
        )

        # Keep the graph's coordinate limits stable.
        self.ax.set_xlim(
            -PLANNER_HALF_WIDTH_M,
            PLANNER_HALF_WIDTH_M,
        )
        self.ax.set_ylim(
            0.0,
            PLANNER_DEPTH_M,
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def render_image(self, target_width: int, target_height: int) -> np.ndarray:
        """Render the current Matplotlib graph as a BGR image."""

        target_width = max(1, int(target_width))
        target_height = max(1, int(target_height))

        self.fig.canvas.draw()

        try:
            rgba = np.asarray(
                self.fig.canvas.buffer_rgba(),
                dtype=np.uint8,
            )
            graph = cv2.cvtColor(
                rgba,
                cv2.COLOR_RGBA2BGR,
            )
        except Exception:
            # Compatibility fallback for older Matplotlib backends.
            rgb = np.frombuffer(
                self.fig.canvas.tostring_rgb(),
                dtype=np.uint8,
            )
            rgb = rgb.reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,)
            )
            graph = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2BGR,
            )

        return cv2.resize(
            graph,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )

    def close(self) -> None:
        try:
            plt.close(self.fig)
        except Exception:
            pass


# ============================================================
# SINGLE-WINDOW DASHBOARD
# ============================================================

def _fit_image(
    image: np.ndarray,
    width: int,
    height: int,
    background: Tuple[int, int, int] = (12, 16, 22),
) -> np.ndarray:
    """Fit an image inside a fixed panel without distortion."""

    width = max(1, int(width))
    height = max(1, int(height))

    panel = np.full(
        (height, width, 3),
        background,
        dtype=np.uint8,
    )

    if image is None or image.size == 0:
        return panel

    ih, iw = image.shape[:2]
    scale = min(
        width / float(max(1, iw)),
        height / float(max(1, ih)),
    )

    nw = max(1, int(round(iw * scale)))
    nh = max(1, int(round(ih * scale)))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_AREA,
    )

    x = (width - nw) // 2
    y = (height - nh) // 2
    panel[y:y + nh, x:x + nw] = resized
    return panel


def draw_dashboard_status(
    panel: np.ndarray,
    plan: Dict[str, Any],
    objects: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    frame_number: int,
) -> np.ndarray:
    """Draw action/warning information in a dedicated visible panel."""

    h, w = panel.shape[:2]
    panel = panel.copy()

    action = str(
        plan.get("action", "KEEP_SPEED")
    ).upper().replace("_", " ")
    direction = str(
        plan.get("direction", "CENTER")
    ).upper()
    warning = str(plan.get("warning", "") or "")

    try:
        speed = float(plan.get("speed", 0.0) or 0.0)
    except (TypeError, ValueError):
        speed = 0.0

    high_risk = sum(
        1 for r in risks
        if str(r.get("risk", "")).upper() == "HIGH_RISK"
    )
    caution = sum(
        1 for r in risks
        if str(r.get("risk", "")).upper() == "CAUTION"
    )

    ttc_values = []
    for risk in risks:
        try:
            value = float(risk.get("ttc"))
            if value >= 0:
                ttc_values.append(value)
        except (TypeError, ValueError):
            pass

    min_ttc = f"{min(ttc_values):.1f}s" if ttc_values else "--"
    status = get_status(plan)

    status_color = {
        "NORMAL": (80, 220, 110),
        "CAUTION": (0, 200, 255),
        "SLOWING": (0, 200, 255),
        "REROUTING": (60, 220, 255),
        "BRAKING": (70, 120, 255),
        "EMERGENCY": (70, 70, 255),
    }.get(status, (255, 255, 255))

    # Title
    put_text(
        panel,
        "DHARMA CONTROL",
        (18, 30),
        0.62,
        (255, 255, 255),
        2,
    )

    put_text(
        panel,
        "LIVE DECISION OUTPUT",
        (18, 50),
        0.34,
        (175, 185, 195),
        1,
    )

    # Large action block
    block_y = 62
    block_h = max(82, int(h * 0.36))
    panel = draw_panel(
        panel,
        12,
        block_y,
        w - 24,
        block_h,
        alpha=0.78,
    )

    put_text(
        panel,
        "ACTION",
        (24, block_y + 24),
        0.34,
        (165, 175, 185),
        1,
    )

    # Fit long actions such as EMERGENCY BRAKE in the narrow panel.
    action_scale = 0.55 if len(action) <= 14 else 0.43
    put_text(
        panel,
        action,
        (24, block_y + 54),
        action_scale,
        status_color,
        2,
    )

    put_text(
        panel,
        f"PATH: {direction}",
        (24, block_y + 78),
        0.38,
        (230, 235, 240),
        1,
    )

    # Warning is deliberately below the action block, never behind the
    # video/player area.
    warn_y = block_y + block_h + 10
    warn_h = max(72, int(h * 0.25))
    panel = draw_panel(
        panel,
        12,
        warn_y,
        w - 24,
        warn_h,
        alpha=0.82,
    )

    put_text(
        panel,
        "WARNING",
        (24, warn_y + 22),
        0.34,
        status_color,
        1,
    )

    warning_text = warning if warning else "NO ACTIVE WARNING"
    if len(warning_text) > 28:
        # Split only for display; the planner value itself is unchanged.
        words = warning_text.split()
        line1 = ""
        line2 = ""
        for word in words:
            if len(line1) + len(word) + 1 <= 24:
                line1 = (line1 + " " + word).strip()
            else:
                line2 = (line2 + " " + word).strip()
        display_lines = [line1, line2]
    else:
        display_lines = [warning_text]

    for i, text in enumerate(display_lines[:2]):
        put_text(
            panel,
            text,
            (24, warn_y + 47 + i * 20),
            0.37,
            status_color if warning else (210, 220, 230),
            1,
        )

    # Compact metrics at the bottom.
    metrics_y = min(
        h - 46,
        warn_y + warn_h + 10,
    )

    put_text(
        panel,
        f"OBJ {len(objects)}   HIGH {high_risk}   CAUTION {caution}",
        (18, metrics_y),
        0.30,
        (215, 220, 225),
        1,
    )
    put_text(
        panel,
        f"SPEED {speed:.1f}   TTC {min_ttc}   FRAME {frame_number}",
        (18, min(h - 8, metrics_y + 20)),
        0.30,
        (215, 220, 225),
        1,
    )

    return panel


def compose_single_window(
    frame: np.ndarray,
    graph_image: np.ndarray,
    plan: Dict[str, Any],
    objects: List[Dict[str, Any]],
    risks: List[Dict[str, Any]],
    frame_number: int,
) -> np.ndarray:
    """Compose video + planner graph + decision status into one window."""

    # Read the actual window size so maximize/restore/resize
    # automatically changes the dashboard layout.
    try:
        _, _, canvas_width, canvas_height = cv2.getWindowImageRect(
            DEMO_WINDOW_TITLE
        )
    except Exception:
        canvas_width = max(
            1200,
            int(round(DEMO_CANVAS_HEIGHT * 16.0 / 9.0)),
        )
        canvas_height = DEMO_CANVAS_HEIGHT

    canvas_width = max(900, int(canvas_width))
    canvas_height = max(500, int(canvas_height))

    left_width = int(canvas_width * DEMO_VIDEO_RATIO)
    right_width = canvas_width - left_width

    video_panel = _fit_image(
        frame,
        left_width,
        canvas_height,
    )

    graph_h = int(canvas_height * DEMO_GRAPH_HEIGHT_RATIO)
    status_h = canvas_height - graph_h

    graph_panel = _fit_image(
        graph_image,
        right_width,
        graph_h,
    )

    status_panel = np.full(
        (status_h, right_width, 3),
        (12, 16, 22),
        dtype=np.uint8,
    )
    status_panel = draw_dashboard_status(
        status_panel,
        plan,
        objects,
        risks,
        frame_number,
    )

    right_panel = np.vstack((graph_panel, status_panel))

    # Thin divider between video and decision/graph information.
    canvas = np.hstack((video_panel, right_panel))
    cv2.line(
        canvas,
        (left_width, 0),
        (left_width, canvas_height - 1),
        (100, 110, 120),
        2,
    )

    return canvas


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    global paused

    cap = cv2.VideoCapture(
        INPUT_VIDEO
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open input video: "
            f"{INPUT_VIDEO}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        fps = 30.0

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (
            width,
            height,
        ),
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            f"Could not create output video: "
            f"{OUTPUT_VIDEO}"
        )

    # --------------------------------------------------------
    # Reset road state
    # --------------------------------------------------------

    reset_road_state()

    # --------------------------------------------------------
    # Initialize modules
    # --------------------------------------------------------

    motion_predictor = MotionPredictor(
        fps=fps
    )

    risk_engine = RiskEngine()

    planner = AdaptivePlanner()

    # Planner graph is rendered into the same OpenCV dashboard window.
    planner_graph = PlannerGraph(
        frame_width=width,
        frame_height=height,
        planner=planner,
    )

    graph_image = np.zeros(
        (
            int(DEMO_CANVAS_HEIGHT * DEMO_GRAPH_HEIGHT_RATIO),
            int(
                max(1200, int(round(DEMO_CANVAS_HEIGHT * 16.0 / 9.0)))
                * DEMO_GRAPH_RATIO
            ),
            3,
        ),
        dtype=np.uint8,
    )

    cv2.namedWindow(
        DEMO_WINDOW_TITLE,
        cv2.WINDOW_NORMAL,
    )

    # Initial size only. The window remains freely resizable/maximizable.
    cv2.resizeWindow(
        DEMO_WINDOW_TITLE,
        1600,
        900,
    )

    frame_number = 0

    print()
    print(
        "=================================================="
    )
    print(
        " DHARMA | FINAL INTEGRATED PIPELINE"
    )
    print(
        "=================================================="
    )
    print(
        f"Input      : {INPUT_VIDEO}"
    )
    print(
        f"Output     : {OUTPUT_VIDEO}"
    )
    print(
        f"Resolution : {width} x {height}"
    )
    print(
        f"FPS        : {fps:.2f}"
    )
    print(
        "--------------------------------------------------"
    )
    print(
        "Processing..."
    )

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        timestamp = (
            frame_number
            / fps
        )

        # ----------------------------------------------------
        # OBJECT DETECTION + TRACKING
        # ----------------------------------------------------

        objects = detect_and_track(
            frame
        )

        # ----------------------------------------------------
        # ROAD UNDERSTANDING
        #
        # Runs in backend.
        # NOTHING from the corridor is drawn.
        # ----------------------------------------------------

        road = detect_drivable_area(
            frame
        )

        # ----------------------------------------------------
        # MOTION PREDICTION
        # ----------------------------------------------------

        predictions = (
            motion_predictor.predict(
                objects
            )
        )

        prediction_map = {
            prediction["id"]:
                prediction.get(
                    "predicted_path",
                    [],
                )
            for prediction in predictions
            if prediction.get(
                "id"
            ) is not None
        }

        # ----------------------------------------------------
        # TEMPORARY PLANNED PATH
        #
        # The planner will replace this conceptually with
        # its selected candidate trajectory.
        #
        # Road center is only used internally for safety
        # assessment.
        # ----------------------------------------------------

        road_center = road.get(
            "road_center",
            [],
        )

        if (
            road_center
            and len(road_center) >= 2
        ):

            planned_path = [
                [
                    float(point[0]),
                    float(point[1]),
                ]
                for point in road_center
            ]

        else:

            planned_path = None

        # ----------------------------------------------------
        # COLLISION RISK
        # ----------------------------------------------------

        risks = risk_engine.calculate_risk(
            objects=objects,
            predictions=prediction_map,
            road=road,
            planned_path=planned_path,
            frame_shape=(
                width,
                height,
            ),
            timestamp=timestamp,
        )

        # ----------------------------------------------------
        # ADAPTIVE PATH PLANNING
        # ----------------------------------------------------

        plan = planner.plan(
            frame_width=width,
            frame_height=height,
            objects=objects,
            risks=risks,
        )

        # ----------------------------------------------------
        # VISUAL OUTPUT
        # ----------------------------------------------------

        # Future object trajectories.
        draw_prediction_paths(
            frame,
            objects,
            predictions,
        )

        # Actual motion vectors.
        prediction_by_id = {
            prediction.get("id"):
                prediction
            for prediction in predictions
            if prediction.get("id") is not None
        }

        for obj in objects:

            object_id = obj.get(
                "id"
            )

            prediction = prediction_by_id.get(
                object_id
            )

            draw_motion_arrow(
                frame,
                obj,
                prediction,
            )

        # Collision risk boxes.
        draw_risks(
            frame,
            objects,
            risks,
        )

        # Selected final trajectory.
        selected_path = plan.get(
            "path",
            None,
        )

        draw_planned_path(
            frame,
            selected_path,
        )

        # Motion graph.
        draw_motion_flow(
            frame,
            objects,
            predictions,
        )

        # ====================================================
        # FINAL STATUS HUD
        # ====================================================

        status = get_status(
            plan
        )

        action = str(
            plan.get(
                "action",
                "KEEP_SPEED",
            )
        ).replace(
            "_",
            " ",
        )

        direction = str(
            plan.get(
                "direction",
                "CENTER",
            )
        ).upper()

        speed = plan.get(
            "speed",
            0.0,
        )

        try:
            speed_value = float(
                speed
            )
        except (
            TypeError,
            ValueError,
        ):
            speed_value = 0.0

        high_risk_count = sum(
            1
            for risk in risks
            if str(
                risk.get(
                    "risk",
                    "",
                )
            ).upper()
            == "HIGH_RISK"
        )

        caution_count = sum(
            1
            for risk in risks
            if str(
                risk.get(
                    "risk",
                    "",
                )
            ).upper()
            == "CAUTION"
        )

        ttc_values = []

        for risk in risks:

            value = risk.get(
                "ttc",
                None,
            )

            if value is None:
                continue

            try:
                value = float(
                    value
                )

                if value >= 0:
                    ttc_values.append(
                        value
                    )
            except (
                TypeError,
                ValueError,
            ):
                continue

        if ttc_values:
            min_ttc_text = (
                f"{min(ttc_values):.1f}s"
            )
        else:
            min_ttc_text = "--"

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        put_text(
            frame,
            "DHARMA",
            (
                20,
                38,
            ),
            0.95,
            (255, 255, 255),
            2,
        )

        put_text(
            frame,
            "AUTONOMOUS SAFETY & PATH ADAPTATION",
            (
                20,
                61,
            ),
            0.40,
            (180, 190, 200),
            1,
        )

        # ----------------------------------------------------
        # Status panel
        # ----------------------------------------------------

        status_w = 285
        status_h = 118

        status_x = 18
        status_y = 82

        frame = draw_panel(
            frame,
            status_x,
            status_y,
            status_w,
            status_h,
            alpha=0.72,
        )

        put_text(
            frame,
            "SYSTEM STATUS",
            (
                status_x + 14,
                status_y + 24,
            ),
            0.40,
            (160, 170, 180),
            1,
        )

        status_color = {
            "NORMAL":
                (80, 220, 110),
            "CAUTION":
                (0, 200, 255),
            "SLOWING":
                (0, 200, 255),
            "REROUTING":
                (60, 220, 255),
            "BRAKING":
                (70, 120, 255),
            "EMERGENCY":
                (70, 70, 255),
        }.get(
            status,
            (255, 255, 255),
        )

        put_text(
            frame,
            status,
            (
                status_x + 14,
                status_y + 53,
            ),
            0.75,
            status_color,
            2,
        )

        put_text(
            frame,
            f"ACTION   {action}",
            (
                status_x + 14,
                status_y + 78,
            ),
            0.42,
            (230, 235, 240),
            1,
        )

        put_text(
            frame,
            f"PATH     {direction}",
            (
                status_x + 14,
                status_y + 99,
            ),
            0.42,
            (230, 235, 240),
            1,
        )

        # ----------------------------------------------------
        # Metrics panel
        # ----------------------------------------------------

        metrics_x = 18
        metrics_y = (
            status_y
            + status_h
            + 12
        )

        metrics_w = 285
        metrics_h = 105

        frame = draw_panel(
            frame,
            metrics_x,
            metrics_y,
            metrics_w,
            metrics_h,
            alpha=0.68,
        )

        put_text(
            frame,
            "LIVE METRICS",
            (
                metrics_x + 14,
                metrics_y + 23,
            ),
            0.40,
            (160, 170, 180),
            1,
        )

        put_text(
            frame,
            f"OBJECTS       {len(objects)}",
            (
                metrics_x + 14,
                metrics_y + 46,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        put_text(
            frame,
            f"HIGH RISK     {high_risk_count}",
            (
                metrics_x + 14,
                metrics_y + 67,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        put_text(
            frame,
            f"CAUTION       {caution_count}",
            (
                metrics_x + 14,
                metrics_y + 88,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        put_text(
            frame,
            f"MIN TTC       {min_ttc_text}",
            (
                metrics_x + 150,
                metrics_y + 46,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        put_text(
            frame,
            f"TARGET SPEED  {speed_value:.1f}",
            (
                metrics_x + 150,
                metrics_y + 67,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        put_text(
            frame,
            f"FRAME         {frame_number}",
            (
                metrics_x + 150,
                metrics_y + 88,
            ),
            0.43,
            (235, 240, 245),
            1,
        )

        # ====================================================
        # WARNING BANNER
        # ====================================================

        warning = str(
            plan.get(
                "warning",
                "",
            )
            or ""
        )

        if warning:

            # Compact warning banner in the TOP-RIGHT.
            # Keeps the road and selected trajectory visible.

            banner_color = status_color

            banner_w = min(
                430,
                max(
                    300,
                    int(width * 0.34),
                ),
            )

            banner_h = 48

            banner_x = (
                width
                - banner_w
                - 18
            )

            banner_y = 18

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (
                    banner_x,
                    banner_y,
                ),
                (
                    width - 18,
                    banner_y + banner_h,
                ),
                (10, 15, 22),
                -1,
            )

            frame = cv2.addWeighted(
                overlay,
                0.84,
                frame,
                0.16,
                0,
            )

            cv2.rectangle(
                frame,
                (
                    banner_x,
                    banner_y,
                ),
                (
                    width - 18,
                    banner_y + banner_h,
                ),
                banner_color,
                2,
            )

            put_text(
                frame,
                warning,
                (
                    banner_x + 16,
                    banner_y + 31,
                ),
                0.52,
                banner_color,
                2,
            )

        # ====================================================
        # SINGLE-WINDOW PLANNER DASHBOARD
        # ====================================================
        # P1-P5 still run on EVERY frame. The Matplotlib graph itself is
        # refreshed every few frames for smoother live playback. The
        # action/warning panel is rendered every frame and is always visible
        # inside the same DHARMA window.
        # ====================================================

        if (
            frame_number % GRAPH_UPDATE_INTERVAL == 0
            or frame_number == 0
        ):
            planner_graph.update(
                road=road,
                objects=objects,
                predictions=predictions,
                risks=risks,
                plan=plan,
                frame_number=frame_number,
            )

            try:
                _, _, win_w, win_h = cv2.getWindowImageRect(
                    DEMO_WINDOW_TITLE
                )
            except Exception:
                win_w = 1600
                win_h = 900

            graph_image = planner_graph.render_image(
                max(300, int(win_w * DEMO_GRAPH_RATIO)),
                max(250, int(win_h * DEMO_GRAPH_HEIGHT_RATIO)),
            )

        dashboard = compose_single_window(
            frame=frame,
            graph_image=graph_image,
            plan=plan,
            objects=objects,
            risks=risks,
            frame_number=frame_number,
        )

        # ----------------------------------------------------
        # PLAY / PAUSE CONTROLS
        # ----------------------------------------------------
        # SPACE pauses on the exact current frame so you can explain
        # the detection, prediction, risk and planner decision.
        # SPACE again resumes processing.

        control_text = (
            "PAUSED  |  SPACE = PLAY"
            if paused
            else "PLAYING  |  SPACE = PAUSE"
        )

        put_text(
            dashboard,
            control_text,
            (18, dashboard.shape[0] - 18),
            0.42,
            (220, 230, 240),
            1,
        )

        put_text(
            dashboard,
            "Q / ESC = EXIT",
            (
                max(18, dashboard.shape[1] - 180),
                dashboard.shape[0] - 18,
            ),
            0.36,
            (175, 185, 195),
            1,
        )

        cv2.imshow(
            DEMO_WINDOW_TITLE,
            dashboard,
        )

        key = cv2.waitKey(
            30 if paused else PLAYBACK_DELAY_MS
        ) & 0xFF

        if key in (ord("q"), 27):
            break

        if key == ord(" "):
            paused = not paused
            continue

        if paused:
            continue

        # ====================================================
        # WRITE
        # ====================================================

        writer.write(
            frame
        )

        frame_number += 1

        if frame_number % 30 == 0:

            print(
                f"Processed "
                f"{frame_number} frames...",
                end="\r",
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    writer.release()

    planner_graph.close()
    cv2.destroyAllWindows()

    print()
    print()
    print(
        "=================================================="
    )
    print(
        " DHARMA FINAL OUTPUT COMPLETE"
    )
    print(
        "=================================================="
    )
    print(
        f"Frames processed : {frame_number}"
    )
    print(
        f"Output video     : {os.path.abspath(OUTPUT_VIDEO)}"
    )
    print(
        "Planner graph    : SAME WINDOW"
    )
    print(
        "Action / warning : ALWAYS VISIBLE IN DASHBOARD"
    )
    print(
        "Road corridor    : BACKEND ONLY"
    )
    print(
        "=================================================="
    )


if __name__ == "__main__":
    main()

    