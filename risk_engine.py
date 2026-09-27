"""
DHARMA - safety/risk_engine.py
================================

Person 4 module: SAFETY & COLLISION RISK

PURPOSE
-------
Consumes tracked-object data (and, when available, motion predictions from
Person 3, drivable-area info from Person 2, and a planned vehicle path from
Person 5) and produces a structured, per-object collision-risk assessment.

This module does NOT:
    - detect or track objects (Person 1's job)
    - segment the road / drivable area (Person 2's job)
    - predict future object trajectories with a learned model (Person 3's job)
    - decide the final steering / vehicle path (Person 5's job)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
TrackedObject = Dict[str, Any]
RiskResult = Dict[str, Any]
TimedPoint = Tuple[Point, float]


# ---------------------------------------------------------------------------
# Small internal data holder for one tracked object's history
# ---------------------------------------------------------------------------

@dataclass
class _ObjectTrack:
    """Rolling history for a single tracked object id."""

    positions: Deque[Tuple[Point, float]] = field(default_factory=deque)
    distances: Deque[Tuple[float, float]] = field(default_factory=deque)
    sizes: Deque[Tuple[float, float]] = field(default_factory=deque)
    last_seen_time: Optional[float] = None


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """
    Configurable collision-risk engine for the DHARMA safety module.

    Combines:

        1. Path-relative collision risk
        2. Visual looming / tau
        3. Ego-corridor proximity
        4. Forward-hazard refinement for motorcycles, bicycles, trucks and buses

    All thresholds are configurable constructor parameters.
    """

    DEFAULT_CLASS_PRIORITY: Dict[str, float] = {
        "person": 1.6,
        "bicycle": 1.4,
        "motorcycle": 1.3,
        "dog": 1.3,
        "cow": 1.3,
        "horse": 1.3,
        "sheep": 1.3,
        "car": 1.0,
        "bus": 1.0,
        "truck": 1.0,
    }

    DEFAULT_PRIORITY = 1.0

    def __init__(
        self,
        history_size: int = 5,
        reaction_time: float = 2.0,
        base_distance: float = 60.0,
        conflict_distance: float = 70.0,
        safe_ttc: float = 6.5,
        caution_ttc: float = 4.0,
        high_risk_ttc: float = 3.0,
        stationary_speed_threshold: float = 2.0,
        lateral_angle_threshold_deg: float = 55.0,
        class_priority: Optional[Dict[str, float]] = None,
        prediction_horizon_seconds: float = 4.0,
        prediction_step_seconds: float = 0.5,
        prediction_dt: float = 0.5,
        ego_corridor_fraction: float = 0.55,
        close_bbox_height_ratio: float = 0.32,
        critical_bbox_height_ratio: float = 0.55,
        close_bbox_absolute_height: float = 180.0,
        critical_bbox_absolute_height: float = 320.0,
    ) -> None:

        self.history_size = max(2, history_size)

        self.reaction_time = reaction_time
        self.base_distance = base_distance
        self.conflict_distance = conflict_distance

        self.safe_ttc = safe_ttc
        self.caution_ttc = caution_ttc
        self.high_risk_ttc = high_risk_ttc

        self.stationary_speed_threshold = stationary_speed_threshold
        self.lateral_angle_threshold_deg = lateral_angle_threshold_deg

        self.prediction_horizon_seconds = max(
            0.5,
            prediction_horizon_seconds,
        )

        self.prediction_step_seconds = max(
            0.1,
            prediction_step_seconds,
        )

        self.prediction_dt = max(
            0.05,
            prediction_dt,
        )

        self.ego_corridor_fraction = min(
            1.0,
            max(0.05, ego_corridor_fraction),
        )

        self.close_bbox_height_ratio = close_bbox_height_ratio
        self.critical_bbox_height_ratio = critical_bbox_height_ratio

        self.close_bbox_absolute_height = close_bbox_absolute_height
        self.critical_bbox_absolute_height = critical_bbox_absolute_height

        self.class_priority: Dict[str, float] = dict(
            self.DEFAULT_CLASS_PRIORITY
        )

        if class_priority:
            self.class_priority.update(class_priority)

        self._tracks: Dict[int, _ObjectTrack] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_risk(
        self,
        objects: Optional[List[TrackedObject]],
        predictions: Optional[Dict[int, List[List[float]]]] = None,
        road: Optional[Any] = None,
        planned_path: Optional[List[List[float]]] = None,
        ego_position: Optional[List[float]] = None,
        frame_shape: Optional[Tuple[int, int]] = None,
        ego_speed: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> List[RiskResult]:

        # Person 2 road data is currently accepted for interface stability.
        _ = road

        if timestamp is None:
            timestamp = time.time()

        if not objects:
            return []

        reference_point = self._resolve_reference_point(
            planned_path,
            ego_position,
        )

        results: List[RiskResult] = []

        # ==============================================================
        # Process every detected object
        # ==============================================================

        for raw_obj in objects:

            obj = self._sanitize_object(raw_obj)

            if obj is None:
                continue

            object_id = obj["id"]
            object_class = obj["class"]
            center = obj["center"]
            bbox = obj["bbox"]

            # ----------------------------------------------------------
            # 1 & 2. Position history + velocity
            # ----------------------------------------------------------

            velocity: Optional[Point] = None

            if object_id is not None:

                velocity = self._update_history_and_get_velocity(
                    object_id,
                    center,
                    timestamp,
                )

            # ----------------------------------------------------------
            # 4. Distance to reference / planned path
            # ----------------------------------------------------------

            distance = self._distance_to_reference(
                center,
                reference_point,
                planned_path,
            )

            # ----------------------------------------------------------
            # 5. Closing speed + current TTC
            # ----------------------------------------------------------

            closing_speed = None

            if (
                object_id is not None
                and distance is not None
            ):

                closing_speed = (
                    self._update_distance_history_and_get_closing_speed(
                        object_id,
                        distance,
                        timestamp,
                    )
                )

            current_ttc = self._time_to_collision(
                distance,
                closing_speed,
            )

            # ----------------------------------------------------------
            # 3. Relative motion classification
            # ----------------------------------------------------------

            relative_motion = self._classify_relative_motion(
                velocity,
                center,
                reference_point,
            )

            # ----------------------------------------------------------
            # 7 & 8. Predicted trajectory + path conflict
            # ----------------------------------------------------------

            trajectory = self._get_or_extrapolate_trajectory(
                object_id,
                center,
                velocity,
                predictions,
            )

            path_conflict, predicted_conflict_ttc = (
                self._check_path_conflict(
                    trajectory,
                    planned_path,
                )
            )

            # ----------------------------------------------------------
            # Visual looming + proximity
            # ----------------------------------------------------------

            visual_ttc: Optional[float] = None

            (
                in_corridor,
                is_close,
                is_critical,
            ) = self._assess_proximity(
                bbox,
                frame_shape,
            )

            if (
                object_id is not None
                and bbox is not None
            ):

                bbox_size = self._bbox_diagonal(bbox)

                visual_ttc = (
                    self._update_size_history_and_get_tau(
                        object_id,
                        bbox_size,
                        timestamp,
                    )
                )

            proximity_conflict = (
                in_corridor
                and (is_close or is_critical)
            )

            # ----------------------------------------------------------
            # Combine all TTC signals
            # ----------------------------------------------------------

            candidate_ttcs = [
                t
                for t in (
                    current_ttc,
                    predicted_conflict_ttc,
                    visual_ttc,
                )
                if t is not None
            ]

            effective_ttc = (
                min(candidate_ttcs)
                if candidate_ttcs
                else None
            )

            # ==========================================================
                        # ==========================================================
            # Base P4 risk classification
            # ==========================================================

            risk = self._classify_risk(
                ttc=effective_ttc,
                path_conflict=path_conflict,
                proximity_conflict=proximity_conflict,
                is_critical=(
                    in_corridor
                    and is_critical
                ),
                distance=distance,
                relative_motion=relative_motion,
                object_class=object_class,
                ego_speed=ego_speed,
            )

            # ----------------------------------------------------------
            # DHARMA COW SAFETY OVERRIDE
            #
            # Any detected cow is always treated as HIGH_RISK,
            # regardless of distance, TTC, motion, or path conflict.
            # ----------------------------------------------------------
            if object_class.strip().lower() == "cow":
                risk = "HIGH_RISK"

            # ==========================================================
            # DHARMA FORWARD-HAZARD REFINEMENT
            #
            # Rule:
            #   - Far objects are treated as HIGH_RISK only when they
            #     are close to the straight-ahead center line.
            #   - Far objects on the left/right side are NOT upgraded
            #     to HIGH_RISK by this refinement.
            #   - Close objects can use the wider forward zone.
            #
            # This is an image-space prototype heuristic.
            # ==========================================================

            if (
                bbox is not None
                and frame_shape is not None
            ):

                frame_w, frame_h = frame_shape

                if (
                    frame_w > 0
                    and frame_h > 0
                ):

                    x1, y1, x2, y2 = bbox

                    bbox_height = max(
                        0.0,
                        y2 - y1,
                    )

                    bbox_cx = (
                        x1 + x2
                    ) / 2.0

                    bbox_bottom = y2

                    # --------------------------------------------------
                    # Normalized image-space measurements
                    # --------------------------------------------------

                    x_from_center = (
                        abs(
                            bbox_cx
                            - (frame_w / 2.0)
                        )
                        / frame_w
                    )

                    height_ratio = (
                        bbox_height
                        / frame_h
                    )

                    bottom_ratio = (
                        bbox_bottom
                        / frame_h
                    )

                    # --------------------------------------------------
                    # Forward viewing zone
                    # --------------------------------------------------

                    forward_zone = (
                        bottom_ratio >= 0.18
                        and bottom_ratio <= 0.72
                    )

                    # Wider zone for nearby objects.
                    forward_lateral_zone = (
                        x_from_center <= 0.34
                    )

                    # Narrow zone for distant objects.
                    # This represents "straight ahead".
                    straight_ahead_zone = (
                        x_from_center <= 0.12
                    )

                    # --------------------------------------------------
                    # TWO-WHEELER
                    # --------------------------------------------------

                    two_wheeler = (
                        object_class.lower()
                        in {
                            "motorcycle",
                            "bicycle",
                            "bike",
                        }
                    )

                    two_wheeler_far = (
                        height_ratio < 0.10
                    )

                    two_wheeler_close = (
                        height_ratio >= 0.045
                        or (
                            distance is not None
                            and distance <= 220.0
                        )
                    )

                    if two_wheeler:

                        # --------------------------------------------------
                        # FAR TWO-WHEELER
                        #
                        # Only straight-ahead distant two-wheelers become
                        # HIGH_RISK.
                        # --------------------------------------------------

                        if (
                            forward_zone
                            and straight_ahead_zone
                            and two_wheeler_far
                        ):

                            risk = "HIGH_RISK"

                        # --------------------------------------------------
                        # CLOSE TWO-WHEELER
                        #
                        # Nearby two-wheelers get a wider forward zone.
                        # --------------------------------------------------

                        elif (
                            forward_zone
                            and forward_lateral_zone
                            and two_wheeler_close
                        ):

                            risk = "HIGH_RISK"

                    # --------------------------------------------------
                    # TRUCK / BUS
                    # --------------------------------------------------

                    large_vehicle = (
                        object_class.lower()
                        in {
                            "truck",
                            "bus",
                        }
                    )

                    large_vehicle_far = (
                        height_ratio < 0.16
                    )

                    large_vehicle_close = (
                        height_ratio >= 0.075
                        or (
                            distance is not None
                            and distance <= 280.0
                        )
                    )

                    if large_vehicle:

                        # --------------------------------------------------
                        # FAR TRUCK / BUS
                        #
                        # Only straight-ahead distant large vehicles get
                        # the additional CAUTION.
                        # --------------------------------------------------

                        if (
                            forward_zone
                            and straight_ahead_zone
                            and large_vehicle_far
                            and large_vehicle_close
                            and risk != "HIGH_RISK"
                        ):

                            risk = "CAUTION"

                        # --------------------------------------------------
                        # CLOSE TRUCK / BUS
                        #
                        # Nearby large vehicles can use the wider zone.
                        # --------------------------------------------------

                        elif (
                            forward_zone
                            and forward_lateral_zone
                            and large_vehicle_close
                            and risk != "HIGH_RISK"
                        ):

                            risk = "CAUTION"

                    # --------------------------------------------------
                    # FAR SIDE-OBJECT PROTECTION
                    #
                    # Do not allow this new refinement to leave a distant
                    # side object as HIGH_RISK simply because it was
                    # visually large enough for another generic signal.
                    #
                    # A distant object must be straight ahead to remain
                    # HIGH_RISK.
                    # --------------------------------------------------

                    if (
                        bottom_ratio >= 0.18
                        and height_ratio < 0.10
                        and not straight_ahead_zone
                        and risk == "HIGH_RISK"
                    ):

                        risk = "CAUTION"

            # ==========================================================
            # ==========================================================
            # Store final result
            # ==========================================================

            results.append(
                {
                    "object_id": object_id,
                    "class": object_class,
                    "risk": risk,
                    "ttc": effective_ttc,
                    "predicted_conflict_ttc": (
                        predicted_conflict_ttc
                    ),
                    "visual_ttc": visual_ttc,
                    "distance": distance,
                    "path_conflict": path_conflict,
                    "proximity_conflict": (
                        proximity_conflict
                    ),
                    "relative_motion": relative_motion,
                }
            )

        # ==============================================================
        # Return results AFTER ALL objects are processed
        # ==============================================================

        return results

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def forget_stale_ids(
        self,
        active_ids: List[int],
    ) -> None:

        active_set = set(active_ids)

        for stale_id in [
            oid
            for oid in self._tracks
            if oid not in active_set
        ]:

            del self._tracks[stale_id]

    # ------------------------------------------------------------------
    # 1. Object sanitation
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_object(
        raw_obj: Any,
    ) -> Optional[TrackedObject]:

        if not isinstance(
            raw_obj,
            dict,
        ):
            return None

        center = raw_obj.get("center")

        if (
            not isinstance(
                center,
                (list, tuple),
            )
            or len(center) != 2
            or not all(
                isinstance(
                    v,
                    (int, float),
                )
                for v in center
            )
        ):
            return None

        object_id = raw_obj.get("id")

        if (
            object_id is not None
            and not isinstance(
                object_id,
                int,
            )
        ):

            try:
                object_id = int(object_id)

            except (
                TypeError,
                ValueError,
            ):

                object_id = None

        object_class = raw_obj.get("class")

        if (
            not isinstance(
                object_class,
                str,
            )
            or not object_class
        ):

            object_class = "unknown"

        bbox_raw = raw_obj.get("bbox")

        bbox: Optional[BBox] = None

        if (
            isinstance(
                bbox_raw,
                (list, tuple),
            )
            and len(bbox_raw) == 4
            and all(
                isinstance(
                    v,
                    (int, float),
                )
                for v in bbox_raw
            )
        ):

            x1, y1, x2, y2 = (
                float(v)
                for v in bbox_raw
            )

            if (
                x2 > x1
                and y2 > y1
            ):

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

        return {
            "id": object_id,
            "class": object_class,
            "center": (
                float(center[0]),
                float(center[1]),
            ),
            "bbox": bbox,
        }

    # ------------------------------------------------------------------
    # 2. Position history + velocity
    # ------------------------------------------------------------------

    def _update_history_and_get_velocity(
        self,
        object_id: int,
        center: Point,
        timestamp: float,
    ) -> Optional[Point]:

        track = self._tracks.setdefault(
            object_id,
            _ObjectTrack(),
        )

        if track.positions.maxlen is None:

            track.positions = deque(
                maxlen=self.history_size
            )

        track.positions.append(
            (
                center,
                timestamp,
            )
        )

        track.last_seen_time = timestamp

        if len(track.positions) < 2:
            return None

        (
            prev_pos,
            prev_t,
        ), (
            curr_pos,
            curr_t,
        ) = (
            track.positions[-2],
            track.positions[-1],
        )

        dt = curr_t - prev_t

        if dt <= 0:
            return None

        vx = (
            curr_pos[0]
            - prev_pos[0]
        ) / dt

        vy = (
            curr_pos[1]
            - prev_pos[1]
        ) / dt

        return (
            vx,
            vy,
        )

    # ------------------------------------------------------------------
    # 3. Relative motion classification
    # ------------------------------------------------------------------

    def _classify_relative_motion(
        self,
        velocity: Optional[Point],
        center: Point,
        reference_point: Optional[Point],
    ) -> str:

        if velocity is None:
            return "unknown"

        speed = math.hypot(
            velocity[0],
            velocity[1],
        )

        if (
            speed
            < self.stationary_speed_threshold
        ):

            return "stationary"

        if reference_point is None:
            return "unknown"

        to_reference = (
            reference_point[0]
            - center[0],
            reference_point[1]
            - center[1],
        )

        ref_dist = math.hypot(
            *to_reference
        )

        if ref_dist == 0:
            return "stationary"

        cos_angle = (
            velocity[0]
            * to_reference[0]
            + velocity[1]
            * to_reference[1]
        ) / (
            speed
            * ref_dist
        )

        cos_angle = max(
            -1.0,
            min(
                1.0,
                cos_angle,
            ),
        )

        angle_deg = math.degrees(
            math.acos(
                cos_angle
            )
        )

        if (
            angle_deg
            <= self.lateral_angle_threshold_deg
        ):

            return "approaching"

        if (
            angle_deg
            >= (
                180
                - self.lateral_angle_threshold_deg
            )
        ):

            return "moving_away"

        return "lateral"

    # ------------------------------------------------------------------
    # 4. Distance estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _euclidean_distance(
        a: Point,
        b: Point,
    ) -> float:

        return math.hypot(
            a[0] - b[0],
            a[1] - b[1],
        )

    @staticmethod
    def _point_to_segment_distance(
        p: Point,
        a: Point,
        b: Point,
    ) -> float:

        ax, ay = a
        bx, by = b
        px, py = p

        abx = bx - ax
        aby = by - ay

        seg_len_sq = (
            abx * abx
            + aby * aby
        )

        if seg_len_sq == 0:

            return math.hypot(
                px - ax,
                py - ay,
            )

        t = (
            (
                px - ax
            ) * abx
            + (
                py - ay
            ) * aby
        ) / seg_len_sq

        t = max(
            0.0,
            min(
                1.0,
                t,
            ),
        )

        closest = (
            ax + t * abx,
            ay + t * aby,
        )

        return math.hypot(
            px - closest[0],
            py - closest[1],
        )

    def _point_to_polyline_distance(
        self,
        p: Point,
        polyline: List[List[float]],
    ) -> float:

        if len(polyline) == 1:

            return self._euclidean_distance(
                p,
                tuple(polyline[0]),
            )

        return min(
            self._point_to_segment_distance(
                p,
                tuple(polyline[i]),
                tuple(polyline[i + 1]),
            )
            for i in range(
                len(polyline) - 1
            )
        )

    def _resolve_reference_point(
        self,
        planned_path: Optional[List[List[float]]],
        ego_position: Optional[List[float]],
    ) -> Optional[Point]:

        if planned_path:
            return tuple(
                planned_path[0]
            )

        if ego_position:
            return tuple(
                ego_position
            )

        return None

    def _distance_to_reference(
        self,
        center: Point,
        reference_point: Optional[Point],
        planned_path: Optional[List[List[float]]],
    ) -> Optional[float]:

        if planned_path:

            return self._point_to_polyline_distance(
                center,
                planned_path,
            )

        if reference_point is not None:

            return self._euclidean_distance(
                center,
                reference_point,
            )

        return None

    # ------------------------------------------------------------------
    # 5. Closing speed + TTC
    # ------------------------------------------------------------------

    def _update_distance_history_and_get_closing_speed(
        self,
        object_id: int,
        distance: float,
        timestamp: float,
    ) -> Optional[float]:

        track = self._tracks.setdefault(
            object_id,
            _ObjectTrack(),
        )

        if track.distances.maxlen is None:

            track.distances = deque(
                maxlen=self.history_size
            )

        track.distances.append(
            (
                distance,
                timestamp,
            )
        )

        if len(track.distances) < 2:
            return None

        (
            prev_d,
            prev_t,
        ), (
            curr_d,
            curr_t,
        ) = (
            track.distances[-2],
            track.distances[-1],
        )

        dt = curr_t - prev_t

        if dt <= 0:
            return None

        return (
            prev_d - curr_d
        ) / dt

    def _time_to_collision(
        self,
        distance: Optional[float],
        closing_speed: Optional[float],
    ) -> Optional[float]:

        if (
            distance is None
            or closing_speed is None
        ):

            return None

        if closing_speed <= 0:
            return None

        return (
            distance
            / closing_speed
        )

    # ------------------------------------------------------------------
    # 6. Safety distance
    # ------------------------------------------------------------------

    def _safety_margin(
        self,
        ego_speed: float,
    ) -> float:

        return (
            self.base_distance
            + self.reaction_time
            * max(
                0.0,
                ego_speed,
            )
        )

    # ------------------------------------------------------------------
    # 7 & 8. Predicted trajectory + path conflict
    # ------------------------------------------------------------------

    def _get_or_extrapolate_trajectory(
        self,
        object_id: Optional[int],
        center: Point,
        velocity: Optional[Point],
        predictions: Optional[
            Dict[int, List[List[float]]]
        ],
    ) -> List[TimedPoint]:

        current: TimedPoint = (
            center,
            0.0,
        )

        # --------------------------------------------------------------
        # Prefer Person 3 prediction
        # --------------------------------------------------------------

        if (
            predictions
            and object_id is not None
            and object_id in predictions
        ):

            raw_traj = predictions[
                object_id
            ]

            if raw_traj:

                timed = [
                    current
                ]

                for i, pt in enumerate(
                    raw_traj,
                    start=1,
                ):

                    t = (
                        i
                        * self.prediction_dt
                    )

                    if (
                        t
                        > self.prediction_horizon_seconds
                    ):

                        break

                    timed.append(
                        (
                            tuple(pt),
                            t,
                        )
                    )

                return timed

        # --------------------------------------------------------------
        # Fallback to constant velocity
        # --------------------------------------------------------------

        if velocity is None:
            return [current]

        timed = [
            current
        ]

        steps = int(
            round(
                self.prediction_horizon_seconds
                / self.prediction_step_seconds
            )
        )

        for i in range(
            1,
            steps + 1,
        ):

            t = (
                i
                * self.prediction_step_seconds
            )

            x = (
                center[0]
                + velocity[0] * t
            )

            y = (
                center[1]
                + velocity[1] * t
            )

            timed.append(
                (
                    (x, y),
                    t,
                )
            )

        return timed

    def _check_path_conflict(
        self,
        trajectory: List[TimedPoint],
        planned_path: Optional[List[List[float]]],
    ) -> Tuple[
        bool,
        Optional[float],
    ]:

        if (
            not planned_path
            or not trajectory
        ):

            return (
                False,
                None,
            )

        earliest_t: Optional[float] = None

        for point, t in trajectory:

            d = (
                self._point_to_polyline_distance(
                    point,
                    planned_path,
                )
            )

            if (
                d
                <= self.conflict_distance
            ):

                if (
                    earliest_t is None
                    or t < earliest_t
                ):

                    earliest_t = t

        return (
            earliest_t is not None,
            earliest_t,
        )

    # ------------------------------------------------------------------
    # Visual looming
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_diagonal(
        bbox: BBox,
    ) -> float:

        x1, y1, x2, y2 = bbox

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        return math.hypot(
            width,
            height,
        )

    def _update_size_history_and_get_tau(
        self,
        object_id: int,
        size: float,
        timestamp: float,
    ) -> Optional[float]:

        track = self._tracks.setdefault(
            object_id,
            _ObjectTrack(),
        )

        if track.sizes.maxlen is None:

            track.sizes = deque(
                maxlen=self.history_size
            )

        track.sizes.append(
            (
                size,
                timestamp,
            )
        )

        if len(track.sizes) < 2:
            return None

        (
            prev_size,
            prev_t,
        ), (
            curr_size,
            curr_t,
        ) = (
            track.sizes[-2],
            track.sizes[-1],
        )

        dt = curr_t - prev_t

        if (
            dt <= 0
            or curr_size <= 0
        ):

            return None

        growth_rate = (
            curr_size - prev_size
        ) / dt

        if growth_rate <= 0:
            return None

        return (
            curr_size
            / growth_rate
        )

    # ------------------------------------------------------------------
    # Proximity assessment
    # ------------------------------------------------------------------

    def _assess_proximity(
        self,
        bbox: Optional[BBox],
        frame_shape: Optional[
            Tuple[int, int]
        ],
    ) -> Tuple[
        bool,
        bool,
        bool,
    ]:

        if bbox is None:

            return (
                False,
                False,
                False,
            )

        x1, y1, x2, y2 = bbox

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        _ = width

        cx = (
            x1 + x2
        ) / 2.0

        if frame_shape:

            frame_w, frame_h = frame_shape

            if (
                frame_w <= 0
                or frame_h <= 0
            ):

                frame_shape = None

        if frame_shape:

            frame_w, frame_h = frame_shape

            corridor_half_width = (
                self.ego_corridor_fraction
                / 2.0
                * frame_w
            )

            frame_center_x = (
                frame_w
                / 2.0
            )

            in_corridor = (
                abs(
                    cx
                    - frame_center_x
                )
                <= corridor_half_width
            )

            height_ratio = (
                height
                / frame_h
            )

            is_close = (
                height_ratio
                >= self.close_bbox_height_ratio
            )

            is_critical = (
                height_ratio
                >= self.critical_bbox_height_ratio
            )

        else:

            in_corridor = True

            is_close = (
                height
                >= self.close_bbox_absolute_height
            )

            is_critical = (
                height
                >= self.critical_bbox_absolute_height
            )

        return (
            in_corridor,
            is_close,
            is_critical,
        )

    # ------------------------------------------------------------------
    # 9 & 10. Risk classification
    # ------------------------------------------------------------------

    def _classify_risk(
        self,
        ttc: Optional[float],
        path_conflict: bool,
        proximity_conflict: bool,
        is_critical: bool,
        distance: Optional[float],
        relative_motion: str,
        object_class: str,
        ego_speed: float,
    ) -> str:

        priority = self.class_priority.get(
            object_class,
            self.DEFAULT_PRIORITY,
        )

        effective_caution_ttc = (
            self.caution_ttc
            * priority
        )

        effective_safe_ttc = (
            self.safe_ttc
            * priority
        )

        margin = self._safety_margin(
            ego_speed
        )

        # --------------------------------------------------------------
        # 1. Critical object
        # --------------------------------------------------------------

        if is_critical:
            return "HIGH_RISK"

        conflict = (
            path_conflict
            or proximity_conflict
        )

        # --------------------------------------------------------------
        # 2. Path/proximity conflict
        # --------------------------------------------------------------

        if conflict:

            if ttc is None:
                return "CAUTION"

            if (
                ttc
                < self.high_risk_ttc
            ):

                return "HIGH_RISK"

            if (
                ttc
                < effective_safe_ttc
            ):

                return "CAUTION"

            return (
                "CAUTION"
                if proximity_conflict
                else "SAFE"
            )

        # --------------------------------------------------------------
        # 3. No conflict
        # --------------------------------------------------------------

        if relative_motion in (
            "moving_away",
            "stationary",
        ):

            if (
                distance is not None
                and distance
                < self.base_distance
            ):

                return "CAUTION"

            return "SAFE"

        # --------------------------------------------------------------
        # 4. Closing object
        # --------------------------------------------------------------

        if (
            ttc is not None
            and ttc
            < effective_caution_ttc
            and distance is not None
            and distance < margin
        ):

            return "CAUTION"

        if (
            distance is not None
            and distance
            < self.base_distance
        ):

            return "CAUTION"

        return "SAFE"