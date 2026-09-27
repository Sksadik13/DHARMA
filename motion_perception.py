from __future__ import annotations

import math
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Tuple


class MotionPredictor:
    """
    Lightweight motion predictor for tracked road objects.

    Uses recent motion history of each ByteTrack ID to estimate
    a stable image-space velocity and extrapolate future position.

    The predictor does not make safety decisions and does not
    plan the ego vehicle path.
    """

    def __init__(
        self,
        fps: float = 30.0,
        history_size: int = 15,
        horizon_sec: float = 2.0,
        num_steps: int = 6,
    ) -> None:

        if fps <= 0:
            raise ValueError("fps must be greater than 0")

        if history_size < 2:
            raise ValueError("history_size must be at least 2")

        if horizon_sec <= 0:
            raise ValueError("horizon_sec must be greater than 0")

        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")

        self.fps = float(fps)
        self.dt = 1.0 / self.fps

        self.history_size = int(history_size)
        self.horizon_sec = float(horizon_sec)
        self.num_steps = int(num_steps)

        # ByteTrack ID -> recent center positions
        self.track_history: Dict[
            int, Deque[Tuple[float, float]]
        ] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

        # ByteTrack ID -> previous smoothed velocity
        self.velocity_history: Dict[
            int, Tuple[float, float]
        ] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_centroid(
        bbox: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate bbox center.

        bbox format:
            [x1, y1, x2, y2]
        """

        if len(bbox) != 4:
            raise ValueError(
                "bbox must contain [x1, y1, x2, y2]"
            )

        x1, y1, x2, y2 = bbox

        return (
            (float(x1) + float(x2)) / 2.0,
            (float(y1) + float(y2)) / 2.0,
        )

    @staticmethod
    def _median(values: List[float]) -> float:
        """
        Return median without requiring NumPy.
        """

        if not values:
            return 0.0

        ordered = sorted(values)
        n = len(ordered)

        middle = n // 2

        if n % 2 == 1:
            return ordered[middle]

        return (
            ordered[middle - 1]
            + ordered[middle]
        ) / 2.0

    def _estimate_velocity(
        self,
        history: Deque[Tuple[float, float]],
    ) -> Tuple[float, float]:
        """
        Estimate velocity using the median of recent
        frame-to-frame velocities.

        Median filtering reduces the effect of noisy
        bounding-box movement and occasional detection jitter.
        """

        if len(history) < 2:
            return 0.0, 0.0

        vx_values: List[float] = []
        vy_values: List[float] = []

        points = list(history)

        # Use only the most recent few movements.
        recent_points = points[-7:]

        for i in range(1, len(recent_points)):
            previous_x, previous_y = recent_points[i - 1]
            current_x, current_y = recent_points[i]

            vx = (
                current_x - previous_x
            ) / self.dt

            vy = (
                current_y - previous_y
            ) / self.dt

            vx_values.append(vx)
            vy_values.append(vy)

        if not vx_values:
            return 0.0, 0.0

        return (
            self._median(vx_values),
            self._median(vy_values),
        )

    # ------------------------------------------------------------------
    # Main prediction interface
    # ------------------------------------------------------------------

    def predict(
        self,
        detected_objects: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Predict future motion of Person 1 tracked objects.

        Parameters
        ----------
        detected_objects:
            Person 1 output:

            [
                {
                    "id": 1,
                    "class": "car",
                    "confidence": 0.94,
                    "bbox": [x1, y1, x2, y2],
                    "center": [cx, cy]
                }
            ]

        Returns
        -------
        List[dict]

        Each prediction contains:

            {
                "id": 1,
                "velocity": [vx, vy],
                "speed": ...,
                "direction": ...,
                "predicted_path": [
                    [x, y],
                    ...
                ],
                "confidence": ...
            }
        """

        predictions: List[Dict[str, Any]] = []

        current_frame_ids = set()

        for obj in detected_objects:

            # ----------------------------------------------------------
            # Validate required Person 1 fields
            # ----------------------------------------------------------

            if "id" not in obj:
                continue

            object_id = obj["id"]

            if object_id is None:
                continue

            object_id = int(object_id)

            if "bbox" not in obj:
                continue

            bbox = obj["bbox"]

            detection_confidence = float(
                obj.get("confidence", 1.0)
            )

            current_frame_ids.add(object_id)

            # ----------------------------------------------------------
            # Current position
            # ----------------------------------------------------------

            cx, cy = self._get_centroid(bbox)

            self.track_history[object_id].append(
                (cx, cy)
            )

            history = self.track_history[object_id]

            num_points = len(history)

            # ----------------------------------------------------------
            # Not enough history
            # ----------------------------------------------------------

            if num_points < 2:

                vx = 0.0
                vy = 0.0

                direction = 0.0
                speed = 0.0

                predicted_path = [
                    [
                        round(cx, 2),
                        round(cy, 2),
                    ]
                    for _ in range(self.num_steps)
                ]

                prediction_confidence = (
                    detection_confidence * 0.3
                )

            else:

                # ------------------------------------------------------
                # Estimate stable recent velocity
                # ------------------------------------------------------

                raw_vx, raw_vy = self._estimate_velocity(
                    history
                )

                # ------------------------------------------------------
                # Smooth velocity over time
                #
                # This prevents sudden direction flips caused by
                # one noisy bounding-box measurement.
                # ------------------------------------------------------

                previous_velocity = (
                    self.velocity_history.get(
                        object_id
                    )
                )

                if previous_velocity is None:

                    vx = raw_vx
                    vy = raw_vy

                else:

                    previous_vx, previous_vy = (
                        previous_velocity
                    )

                    smoothing_factor = 0.35

                    vx = (
                        smoothing_factor * raw_vx
                        + (1.0 - smoothing_factor)
                        * previous_vx
                    )

                    vy = (
                        smoothing_factor * raw_vy
                        + (1.0 - smoothing_factor)
                        * previous_vy
                    )

                # ------------------------------------------------------
                # Small-motion suppression
                #
                # Prevents tiny detection jitter from becoming
                # a fake movement direction.
                # ------------------------------------------------------

                speed_before_filter = math.sqrt(
                    vx * vx + vy * vy
                )

                if speed_before_filter < 8.0:
                    vx = 0.0
                    vy = 0.0

                self.velocity_history[object_id] = (
                    vx,
                    vy,
                )

                # ------------------------------------------------------
                # Speed
                # ------------------------------------------------------

                speed = math.sqrt(
                    vx * vx + vy * vy
                )

                # ------------------------------------------------------
                # Direction
                # ------------------------------------------------------

                if speed <= 0.0:

                    direction = 0.0

                else:

                    direction = (
                        math.degrees(
                            math.atan2(vy, vx)
                        )
                        % 360.0
                    )

                # ------------------------------------------------------
                # Future trajectory
                # ------------------------------------------------------

                step_dt = (
                    self.horizon_sec
                    / self.num_steps
                )

                predicted_path: List[List[float]] = []

                for step in range(
                    1,
                    self.num_steps + 1,
                ):

                    future_t = (
                        step * step_dt
                    )

                    future_x = (
                        cx + vx * future_t
                    )

                    future_y = (
                        cy + vy * future_t
                    )

                    predicted_path.append(
                        [
                            round(future_x, 2),
                            round(future_y, 2),
                        ]
                    )

                # ------------------------------------------------------
                # Confidence
                # ------------------------------------------------------

                history_factor = min(
                    1.0,
                    num_points
                    / float(self.history_size),
                )

                prediction_confidence = (
                    detection_confidence
                    * history_factor
                )

            # ----------------------------------------------------------
            # Store prediction
            # ----------------------------------------------------------

            predictions.append(
                {
                    "id": object_id,

                    "velocity": [
                        round(vx, 2),
                        round(vy, 2),
                    ],

                    "speed": round(
                        speed,
                        2,
                    ),

                    "direction": round(
                        direction,
                        2,
                    ),

                    "predicted_path":
                        predicted_path,

                    "confidence": round(
                        max(
                            0.0,
                            min(
                                1.0,
                                prediction_confidence,
                            ),
                        ),
                        2,
                    ),
                }
            )

        # --------------------------------------------------------------
        # Remove disappeared objects
        # --------------------------------------------------------------

        stale_ids = (
            set(self.track_history.keys())
            - current_frame_ids
        )

        for stale_id in stale_ids:

            del self.track_history[stale_id]

            if stale_id in self.velocity_history:
                del self.velocity_history[stale_id]

        return predictions

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Clear all motion history.

        Call this when switching to a new video or scenario.
        """

        self.track_history.clear()
        self.velocity_history.clear()


# ----------------------------------------------------------------------
# Simple functional interface
# ----------------------------------------------------------------------

_default_predictor: MotionPredictor | None = None


def get_predictor(
    fps: float = 30.0,
) -> MotionPredictor:
    """
    Return the shared predictor instance.
    """

    global _default_predictor

    if _default_predictor is None:

        _default_predictor = MotionPredictor(
            fps=fps
        )

    return _default_predictor


def predict_motion(
    detected_objects: List[Dict[str, Any]],
    fps: float = 30.0,
) -> List[Dict[str, Any]]:
    """
    Module-level convenience function.
    """

    predictor = get_predictor(
        fps=fps
    )

    return predictor.predict(
        detected_objects
    )


def reset_motion_state() -> None:
    """
    Reset the shared predictor history.
    """

    global _default_predictor

    if _default_predictor is not None:

        _default_predictor.reset()