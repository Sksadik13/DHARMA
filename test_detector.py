"""
Tests for perception.detector (Module 1: Object Detection & Tracking).

Run with:
    cd Dharma
    python -m pytest tests/ -v

These are integration-style tests (they load real YOLO11n weights and run
real inference) rather than pure unit tests, since the whole point of this
module is the detector+tracker behavior. They're kept fast by using small,
short inputs.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perception.detector import ObjectTracker, detect_and_track  # noqa: E402

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
ULTRALYTICS_SAMPLE_IMG = Path(
    __import__("ultralytics").__file__
).parent / "assets" / "bus.jpg"


@pytest.fixture(scope="module")
def tracker() -> ObjectTracker:
    return ObjectTracker(verbose=False)


def test_output_schema_on_real_image(tracker: ObjectTracker) -> None:
    """detect_and_track must return the agreed-upon structured format."""
    frame = cv2.imread(str(ULTRALYTICS_SAMPLE_IMG))
    objects = tracker.detect_and_track(frame)

    assert isinstance(objects, list)
    assert len(objects) > 0, "expected at least one detection on bus.jpg"

    for obj in objects:
        assert set(obj.keys()) == {"id", "class", "confidence", "bbox", "center"}
        assert isinstance(obj["class"], str)
        assert 0.0 <= obj["confidence"] <= 1.0
        assert len(obj["bbox"]) == 4
        x1, y1, x2, y2 = obj["bbox"]
        assert x1 < x2 and y1 < y2, "bbox should be [x1, y1, x2, y2] with x1<x2, y1<y2"
        assert len(obj["center"]) == 2
        cx, cy = obj["center"]
        assert x1 <= cx <= x2
        assert y1 <= cy <= y2


def test_detects_expected_classes(tracker: ObjectTracker) -> None:
    """bus.jpg contains a bus and several pedestrians -- sanity-check the
    class vocabulary coming out matches what's actually in the scene."""
    frame = cv2.imread(str(ULTRALYTICS_SAMPLE_IMG))
    objects = tracker.detect_and_track(frame)
    classes_found = {obj["class"] for obj in objects}

    assert "person" in classes_found
    assert "bus" in classes_found


def test_blank_frame_returns_empty_list(tracker: ObjectTracker) -> None:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    objects = tracker.detect_and_track(blank)
    assert objects == []


def test_none_frame_raises(tracker: ObjectTracker) -> None:
    with pytest.raises(ValueError):
        tracker.detect_and_track(None)


def test_module_level_convenience_function() -> None:
    """The bare `detect_and_track(frame)` import should also work, per the
    interface sketched in the project spec."""
    frame = cv2.imread(str(ULTRALYTICS_SAMPLE_IMG))
    objects = detect_and_track(frame)
    assert isinstance(objects, list)
    assert len(objects) > 0


def test_tracking_ids_persist_across_frames(tracker: ObjectTracker) -> None:
    """The whole point of ByteTrack: the same physical object should keep
    the same ID across consecutive frames of a video."""
    video_path = ASSETS_DIR / "synthetic_test.mp4"
    if not video_path.exists():
        pytest.skip(
            "synthetic_test.mp4 not found -- run "
            "assets/make_synthetic_video.py first to generate it."
        )

    id_sets_per_frame = [
        {obj["id"] for obj in objects}
        for objects in tracker.process_video(video_path)
    ]
    assert len(id_sets_per_frame) > 5, "expected multiple frames"

    # every ID seen in the first frame should still be present a few
    # frames later (short synthetic clip, no occlusion -- IDs shouldn't
    # churn)
    first_frame_ids = id_sets_per_frame[0]
    later_frame_ids = id_sets_per_frame[5]
    assert first_frame_ids & later_frame_ids, (
        f"expected overlapping track IDs between frame 0 ({first_frame_ids}) "
        f"and frame 5 ({later_frame_ids}) -- tracker identity is not persisting"
    )


def test_process_video_missing_file_raises(tracker: ObjectTracker) -> None:
    with pytest.raises(FileNotFoundError):
        list(tracker.process_video("does_not_exist.mp4"))
