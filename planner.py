"""
DHARMA
Dynamic Hazard-aware Autonomous Routing and Motion Adaptation

Person 5:
Adaptive Path Planning

Risk-aware path planning, alternate path selection,
speed adaptation, braking and emergency braking.
"""

from dataclasses import dataclass


@dataclass
class PlannerConfig:
    """Configuration for the adaptive path planner."""

    # --------------------------------------------------------
    # Candidate paths
    # --------------------------------------------------------

    num_candidates: int = 5

    max_lateral_offset: float = 120.0

    safety_margin: float = 35.0

    # --------------------------------------------------------
    # TTC thresholds
    # --------------------------------------------------------

    caution_ttc: float = 4.0

    slow_down_ttc: float = 3.0

    brake_ttc: float = 2.0

    emergency_ttc: float = 1.0

    # --------------------------------------------------------
    # CLOSE OBJECT THRESHOLD
    #
    # Object bottom position is measured as a fraction
    # of image height.
    #
    # Example:
    #
    # 0.85 = object reaches 85% of image height
    #
    # Higher value means the object is closer to ego.
    # --------------------------------------------------------

    emergency_close_ratio: float = 0.78

    # Minimum object size required for emergency braking.
    #
    # This prevents tiny/far objects from triggering
    # emergency braking.
    emergency_min_height_ratio: float = 0.10

    # Emergency object must also be reasonably close
    # to the image center.
    emergency_center_ratio: float = 0.25

    # --------------------------------------------------------
    # Speed recommendations
    # --------------------------------------------------------

    normal_speed: float = 40.0

    caution_speed: float = 30.0

    slow_speed: float = 20.0

    brake_speed: float = 5.0

    emergency_speed: float = 0.0


class AdaptivePlanner:
    """
    Adaptive path planner.

    Possible actions:

        KEEP_SPEED
        CAUTION
        SLOW_DOWN
        REROUTE
        BRAKE
        EMERGENCY_BRAKE
    """

    def __init__(
        self,
        config: PlannerConfig | None = None,
    ):

        self.config = (
            config
            if config is not None
            else PlannerConfig()
        )

    # ========================================================
    # CANDIDATE PATH GENERATION
    # ========================================================

    def generate_candidate_paths(
        self,
        frame_width: int,
        frame_height: int,
    ):
        """
        Generate multiple candidate paths.

        Five paths are generated:

            LEFT
            LEFT
            CENTER
            RIGHT
            RIGHT
        """

        center_x = frame_width / 2.0

        bottom_y = frame_height * 0.92

        top_y = frame_height * 0.45

        offsets = [
            -self.config.max_lateral_offset,
            -self.config.max_lateral_offset * 0.5,
            0.0,
            self.config.max_lateral_offset * 0.5,
            self.config.max_lateral_offset,
        ]

        candidates = []

        num_points = 20

        for offset in offsets:

            path = []

            for i in range(num_points):

                t = i / float(
                    num_points - 1
                )

                perspective = (
                    1.0
                    - 0.65 * t
                )

                x = (
                    center_x
                    + offset
                    * perspective
                )

                y = (
                    bottom_y
                    + (
                        top_y
                        - bottom_y
                    )
                    * t
                )

                path.append(
                    [
                        float(x),
                        float(y),
                    ]
                )

            if offset < 0:

                direction = "LEFT"

            elif offset > 0:

                direction = "RIGHT"

            else:

                direction = "CENTER"

            candidates.append(
                {
                    "direction": direction,
                    "offset": float(offset),
                    "path": path,
                }
            )

        return candidates

    # ========================================================
    # PATH COLLISION CHECK
    # ========================================================

    def check_path_collision(
        self,
        path,
        objects,
    ):
        """
        Check whether a candidate path comes too close
        to detected objects.
        """

        if not path:

            return {
                "collision": False,
                "obstacles": [],
                "min_clearance": float(
                    "inf"
                ),
            }

        min_clearance = float(
            "inf"
        )

        obstacles = []

        for obj in objects:

            bbox = obj.get(
                "bbox"
            )

            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = map(
                float,
                bbox,
            )

            object_x = (
                x1 + x2
            ) / 2.0

            object_y = (
                y1 + y2
            ) / 2.0

            object_width = max(
                1.0,
                x2 - x1,
            )

            object_height = max(
                1.0,
                y2 - y1,
            )

            object_radius = max(
                object_width,
                object_height,
            ) / 2.0

            path_clearance = float(
                "inf"
            )

            for point in path:

                px = float(
                    point[0]
                )

                py = float(
                    point[1]
                )

                distance = (
                    (
                        px
                        - object_x
                    ) ** 2
                    +
                    (
                        py
                        - object_y
                    ) ** 2
                ) ** 0.5

                distance -= (
                    object_radius
                )

                path_clearance = min(
                    path_clearance,
                    distance,
                )

            min_clearance = min(
                min_clearance,
                path_clearance,
            )

            if (
                path_clearance
                < self.config.safety_margin
            ):

                obstacles.append(
                    {
                        "id": obj.get(
                            "id"
                        ),
                        "class": obj.get(
                            "class",
                            "unknown",
                        ),
                        "clearance": float(
                            path_clearance
                        ),
                    }
                )

        return {
            "collision": (
                len(obstacles) > 0
            ),
            "obstacles": obstacles,
            "min_clearance": float(
                min_clearance
            ),
        }

    # ========================================================
    # PATH SELECTION
    # ========================================================

    def select_safest_path(
        self,
        candidates,
        objects,
    ):
        """
        Evaluate all candidate paths and select
        the safest available trajectory.
        """

        if not candidates:

            return {
                "selected": None,
                "direction": "NONE",
                "path": [],
                "collision": True,
                "clearance": 0.0,
                "obstacles": [],
                "action": "BRAKE",
                "target_speed": (
                    self.config.brake_speed
                ),
                "warning": (
                    "BRAKING — "
                    "NO PATH AVAILABLE"
                ),
                "reason": (
                    "No candidate paths available"
                ),
            }

        evaluated = []

        for candidate in candidates:

            result = (
                self.check_path_collision(
                    candidate["path"],
                    objects,
                )
            )

            evaluated.append(
                {
                    "candidate": candidate,
                    "collision": result[
                        "collision"
                    ],
                    "clearance": result[
                        "min_clearance"
                    ],
                    "obstacles": result[
                        "obstacles"
                    ],
                }
            )

        safe_paths = [
            item
            for item in evaluated
            if not item["collision"]
        ]

        if safe_paths:

            safe_paths.sort(
                key=lambda item: (
                    item["clearance"],
                    -abs(
                        item[
                            "candidate"
                        ]["offset"]
                    ),
                ),
                reverse=True,
            )

            selected = safe_paths[0]

            candidate = selected[
                "candidate"
            ]

            direction = candidate[
                "direction"
            ]

            if direction == "CENTER":

                action = "KEEP_SPEED"

                target_speed = (
                    self.config.normal_speed
                )

                warning = "PATH CLEAR"

                reason = (
                    "Center path is clear"
                )

            else:

                action = "REROUTE"

                target_speed = (
                    self.config.caution_speed
                )

                warning = (
                    "ALTERNATE PATH — "
                    f"{direction}"
                )

                reason = (
                    "Center path blocked; "
                    f"safe {direction.lower()} "
                    "path selected"
                )

            return {
                "selected": candidate,
                "direction": direction,
                "path": candidate[
                    "path"
                ],
                "collision": False,
                "clearance": selected[
                    "clearance"
                ],
                "obstacles": selected[
                    "obstacles"
                ],
                "action": action,
                "target_speed": target_speed,
                "warning": warning,
                "reason": reason,
            }

        evaluated.sort(
            key=lambda item: item[
                "clearance"
            ],
            reverse=True,
        )

        best = evaluated[0]

        return {
            "selected": best[
                "candidate"
            ],
            "direction": "NONE",
            "path": best[
                "candidate"
            ]["path"],
            "collision": True,
            "clearance": best[
                "clearance"
            ],
            "obstacles": best[
                "obstacles"
            ],
            "action": "BRAKE",
            "target_speed": (
                self.config.brake_speed
            ),
            "warning": (
                "BRAKING — "
                "PATH BLOCKED"
            ),
            "reason": (
                "No collision-free "
                "alternate path available"
            ),
        }

    # ========================================================
    # CLOSE OBJECT CHECK
    # ========================================================

    def is_close_emergency_object(
        self,
        risk,
        objects,
        frame_width,
        frame_height,
    ):
        """
        Determine whether a risk belongs to a genuinely
        close object that is capable of triggering
        emergency braking.

        Emergency braking requires:

            1. Valid object
            2. Object is close vertically
            3. Object is large enough
            4. Object is reasonably near the ego path
        """

        object_id = risk.get(
            "object_id"
        )

        if object_id is None:
            return False

        target = None

        for obj in objects:

            if obj.get("id") == object_id:

                target = obj
                break

        if target is None:
            return False

        bbox = target.get(
            "bbox"
        )

        if not bbox or len(bbox) < 4:
            return False

        x1, y1, x2, y2 = map(
            float,
            bbox,
        )

        object_width = (
            x2 - x1
        )

        object_height = (
            y2 - y1
        )

        if object_height <= 0:
            return False

        # ----------------------------------------------------
        # Object bottom position
        #
        # A larger bottom ratio means the object is
        # visually closer to the ego vehicle.
        # ----------------------------------------------------

        bottom_ratio = (
            y2
            / float(frame_height)
        )

        close_enough = (
            bottom_ratio
            >= self.config.emergency_close_ratio
        )

        if not close_enough:
            return False

        # ----------------------------------------------------
        # Object size
        # ----------------------------------------------------

        height_ratio = (
            object_height
            / float(frame_height)
        )

        large_enough = (
            height_ratio
            >= self.config.emergency_min_height_ratio
        )

        if not large_enough:
            return False

        # ----------------------------------------------------
        # Horizontal position
        #
        # Emergency objects should be reasonably close
        # to the ego vehicle's forward corridor.
        # ----------------------------------------------------

        object_center_x = (
            x1 + x2
        ) / 2.0

        horizontal_offset = abs(
            object_center_x
            - frame_width / 2.0
        )

        horizontal_ratio = (
            horizontal_offset
            / float(frame_width)
        )

        in_forward_zone = (
            horizontal_ratio
            <= self.config.emergency_center_ratio
        )

        if not in_forward_zone:
            return False

        return True

    # ========================================================
    # RISK EVALUATION
    # ========================================================

    def evaluate_risk(
        self,
        risks,
        objects=None,
        frame_width=None,
        frame_height=None,
    ):
        """
        Evaluate P4 risk results.

        Emergency braking is intentionally restricted
        to close objects in the forward driving zone.
        """

        if risks is None:
            risks = []

        if objects is None:
            objects = []

        # ----------------------------------------------------
        # If image dimensions are unavailable, use
        # conservative non-emergency behavior.
        # ----------------------------------------------------

        dimensions_available = (
            frame_width is not None
            and frame_height is not None
            and frame_width > 0
            and frame_height > 0
        )

        highest_risk = "SAFE"

        lowest_ttc = float(
            "inf"
        )

        emergency_threat = None

        risk_priority = {
            "SAFE": 0,
            "CAUTION": 1,
            "HIGH_RISK": 2,
        }

        # ----------------------------------------------------
        # Inspect every risk
        # ----------------------------------------------------

        for risk in risks:

            risk_level = str(
                risk.get(
                    "risk",
                    "SAFE",
                )
            ).upper()

            if (
                risk_priority.get(
                    risk_level,
                    0,
                )
                >
                risk_priority.get(
                    highest_risk,
                    0,
                )
            ):

                highest_risk = (
                    risk_level
                )

            ttc_values = []

            # Primary TTC
            ttc = risk.get(
                "ttc"
            )

            if ttc is not None:

                try:

                    ttc_values.append(
                        float(ttc)
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            # Predicted conflict TTC
            predicted_ttc = risk.get(
                "predicted_conflict_ttc"
            )

            if predicted_ttc is not None:

                try:

                    ttc_values.append(
                        float(
                            predicted_ttc
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):
                    pass

            if ttc_values:

                object_ttc = min(
                    ttc_values
                )

                lowest_ttc = min(
                    lowest_ttc,
                    object_ttc,
                )

                # --------------------------------------------
                # Emergency candidate
                #
                # IMPORTANT:
                # A low TTC alone is NOT enough.
                # The object must also be physically close.
                # --------------------------------------------

                if (
                    object_ttc
                    <= self.config.emergency_ttc
                    and dimensions_available
                ):

                    is_close = (
                        self.is_close_emergency_object(
                            risk=risk,
                            objects=objects,
                            frame_width=frame_width,
                            frame_height=frame_height,
                        )
                    )

                    if is_close:

                        emergency_threat = risk

        # ----------------------------------------------------
        # Emergency brake
        # ----------------------------------------------------

        if emergency_threat is not None:

            emergency_ttc = emergency_threat.get(
                "ttc"
            )

            return {
                "action": "EMERGENCY_BRAKE",
                "target_speed": (
                    self.config.emergency_speed
                ),
                "warning": (
                    "EMERGENCY BRAKE — "
                    "COLLISION RISK"
                ),
                "direction": "NONE",
                "reason": (
                    "Close forward obstacle "
                    f"with critical TTC: "
                    f"{float(emergency_ttc):.2f}s"
                ),
            }

        # ----------------------------------------------------
        # Normal brake
        #
        # A non-emergency brake can still happen because
        # of TTC/path blockage.
        # ----------------------------------------------------

        if (
            lowest_ttc
            <= self.config.brake_ttc
        ):

            return {
                "action": "BRAKE",
                "target_speed": (
                    self.config.brake_speed
                ),
                "warning": (
                    "BRAKING — "
                    "COLLISION RISK"
                ),
                "direction": "NONE",
                "reason": (
                    f"Low TTC: "
                    f"{lowest_ttc:.2f}s"
                ),
            }

        # ----------------------------------------------------
        # Slow down
        # ----------------------------------------------------

        if (
            lowest_ttc
            <= self.config.slow_down_ttc
            or highest_risk == "HIGH_RISK"
        ):

            return {
                "action": "SLOW_DOWN",
                "target_speed": (
                    self.config.slow_speed
                ),
                "warning": (
                    "SLOW DOWN — "
                    "HAZARD AHEAD"
                ),
                "direction": "CENTER",
                "reason": (
                    f"Risk: "
                    f"{highest_risk}"
                ),
            }

        # ----------------------------------------------------
        # Caution
        # ----------------------------------------------------

        if (
            lowest_ttc
            <= self.config.caution_ttc
            or highest_risk == "CAUTION"
        ):

            return {
                "action": "CAUTION",
                "target_speed": (
                    self.config.caution_speed
                ),
                "warning": (
                    "CAUTION — "
                    "OBSTACLE DETECTED"
                ),
                "direction": "CENTER",
                "reason": (
                    f"Risk: "
                    f"{highest_risk}"
                ),
            }

        # ----------------------------------------------------
        # Safe
        # ----------------------------------------------------

        return {
            "action": "KEEP_SPEED",
            "target_speed": (
                self.config.normal_speed
            ),
            "warning": "",
            "direction": "CENTER",
            "reason": (
                "Risk level acceptable"
            ),
        }

    # ========================================================
    # COMPLETE PLANNING
    # ========================================================

    def plan(
        self,
        frame_width,
        frame_height,
        objects,
        risks=None,
    ):
        """
        Complete adaptive planning decision.
        """

        if risks is None:
            risks = []

        # ----------------------------------------------------
        # Generate candidate trajectories
        # ----------------------------------------------------

        candidates = (
            self.generate_candidate_paths(
                frame_width,
                frame_height,
            )
        )

        # ----------------------------------------------------
        # Evaluate risk
        # ----------------------------------------------------

        risk_decision = (
            self.evaluate_risk(
                risks=risks,
                objects=objects,
                frame_width=frame_width,
                frame_height=frame_height,
            )
        )

        # ----------------------------------------------------
        # Emergency brake
        #
        # This now only happens when a CLOSE object
        # satisfies the emergency conditions.
        # ----------------------------------------------------

        if (
            risk_decision["action"]
            == "EMERGENCY_BRAKE"
        ):

            return {
                "action": (
                    "EMERGENCY_BRAKE"
                ),
                "direction": "NONE",
                "path": [],
                "target_speed": (
                    self.config.emergency_speed
                ),
                "warning": (
                    "EMERGENCY BRAKE — "
                    "COLLISION RISK"
                ),
                "reason": risk_decision[
                    "reason"
                ],
            }

        # ----------------------------------------------------
        # Evaluate candidate paths
        # ----------------------------------------------------

        path_decision = (
            self.select_safest_path(
                candidates,
                objects,
            )
        )

        # ----------------------------------------------------
        # No collision-free path
        # ----------------------------------------------------

        if path_decision["collision"]:

            return {
                "action": "BRAKE",
                "direction": "NONE",
                "path": [],
                "target_speed": (
                    self.config.brake_speed
                ),
                "warning": (
                    "BRAKING — "
                    "PATH BLOCKED"
                ),
                "reason": (
                    path_decision[
                        "reason"
                    ]
                ),
            }

        # ----------------------------------------------------
        # Alternate path
        # ----------------------------------------------------

        if (
            path_decision["direction"]
            != "CENTER"
        ):

            return {
                "action": "REROUTE",
                "direction": (
                    path_decision[
                        "direction"
                    ]
                ),
                "path": (
                    path_decision["path"]
                ),
                "target_speed": (
                    self.config.caution_speed
                ),
                "warning": (
                    "ALTERNATE PATH — "
                    f"{path_decision['direction']}"
                ),
                "reason": (
                    path_decision[
                        "reason"
                    ]
                ),
            }

        # ----------------------------------------------------
        # Slow down
        # ----------------------------------------------------

        if (
            risk_decision["action"]
            == "SLOW_DOWN"
        ):

            return {
                "action": "SLOW_DOWN",
                "direction": "CENTER",
                "path": (
                    path_decision["path"]
                ),
                "target_speed": (
                    self.config.slow_speed
                ),
                "warning": (
                    "SLOW DOWN — "
                    "HAZARD AHEAD"
                ),
                "reason": (
                    risk_decision[
                        "reason"
                    ]
                ),
            }

        # ----------------------------------------------------
        # Caution
        # ----------------------------------------------------

        if (
            risk_decision["action"]
            == "CAUTION"
        ):

            return {
                "action": "CAUTION",
                "direction": "CENTER",
                "path": (
                    path_decision["path"]
                ),
                "target_speed": (
                    self.config.caution_speed
                ),
                "warning": (
                    "CAUTION — "
                    "OBSTACLE DETECTED"
                ),
                "reason": (
                    risk_decision[
                        "reason"
                    ]
                ),
            }

        # ----------------------------------------------------
        # Normal driving
        # ----------------------------------------------------

        return {
            "action": "KEEP_SPEED",
            "direction": "CENTER",
            "path": (
                path_decision["path"]
            ),
            "target_speed": (
                self.config.normal_speed
            ),
            "warning": "",
            "reason": (
                "Center path is clear"
            ),
        }