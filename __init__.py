"""DHARMA - Person 2 package: road & drivable area detection."""

from .drivable_area import (
    DrivableAreaDetector,
    RoadConfig,
    backend_info,
    configure,
    detect_drivable_area,
    get_detector,
    reset_state,
)

from .corridor_overlay import (
    draw_corridor,
    make_mask_view,
    stack_side_by_side,
)

__all__ = [
    "detect_drivable_area",
    "DrivableAreaDetector",
    "RoadConfig",
    "configure",
    "reset_state",
    "get_detector",
    "backend_info",
    "draw_corridor",
    "make_mask_view",
    "stack_side_by_side",
]