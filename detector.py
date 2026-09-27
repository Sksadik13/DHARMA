"""
Dharma - Module 1: Object Detection & Tracking
================================================

Answers: "What objects are present in the scene, where are they, what class
are they, and which detections belong to the same object across frames?"

This module deliberately does NOT decide whether an object is dangerous and
does NOT plan the vehicle's path -- that is the job of Modules 3, 4 and 5.
It is the first stage of the Dharma perception -> prediction -> safety ->
planning pipeline, and its only contract with the rest of the system is the
stable output format documented below.

Pipeline implemented here:

    Frame -> YOLO11n (detection) -> ByteTrack (identity across frames)
          -> structured object list

Usage
-----
    from perception.detector import ObjectTracker

    tracker = ObjectTracker()
    objects = tracker.detect_and_track(frame)   # frame: BGR np.ndarray (as from cv2)

Each element of `objects` is a dict:

    {
        "id": 1,                       # persistent tracking ID (int) or None if untracked
        "class": "car",                # human-readable COCO class name
        "confidence": 0.94,            # detection confidence, float in [0, 1]
        "bbox": [120, 250, 300, 500],  # [x1, y1, x2, y2] in pixel coordinates
        "center": [210, 375],          # [cx, cy] in pixel coordinates
    }

A module-level convenience function `detect_and_track(frame)` is also
provided (backed by a lazily-created default ObjectTracker instance) to
match the interface sketched in the project spec exactly:

    from perception.detector import detect_and_track
    objects = detect_and_track(frame)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger("dharma.perception.detector")

# ---------------------------------------------------------------------------
# Objects of interest
# ---------------------------------------------------------------------------
# The pretrained YOLO11n model ships trained on COCO's 80 classes. We restrict
# inference to the subset that is actually relevant to a road scene, both to
# cut down false positives (no point detecting "toaster") and to keep the
# downstream modules' class vocabulary small and predictable.
#
# IMPORTANT (see "Known Coverage Gaps" below): COCO does NOT contain classes
# for "auto-rickshaw", "pushcart", or generic "cattle". We do not rename or
# remap detections into these labels -- that would silently fabricate data
# the model never actually produced. Instead we surface the closest COCO
# class as-is and document the gap so the team can decide whether to swap in
# a different pretrained model or fine-tune later.

COCO_CLASSES_OF_INTEREST = {
    # Vehicles
    "car",
    "motorcycle",
    "bus",
    "truck",
    # Vulnerable road users
    "person",
    "bicycle",
    # Animals (closest COCO proxies for Indian road fauna -- see notes below)
    "cow",     # closest available proxy for "cattle"
    "horse",
    "dog",
    "sheep",
}

# Human-readable notes on Indian-road classes that are NOT reliably covered
# by stock COCO-pretrained YOLO11n. Exposed so calling code / docs can surface
# this to the team rather than silently pretending coverage exists.
KNOWN_COVERAGE_GAPS = {
    "auto-rickshaw": (
        "No dedicated COCO class. Auto-rickshaws are sometimes picked up as "
        "'car' or missed entirely depending on angle/occlusion. Needs a "
        "custom-trained or fine-tuned class if reliable auto-rickshaw "
        "detection is required."
    ),
    "pushcart": (
        "No dedicated COCO class. Not reliably detected by stock YOLO11n; "
        "may be missed, or fragments may be picked up as unrelated objects. "
        "Needs fine-tuning or a different pretrained model."
    ),
    "cattle": (
        "No 'cattle' class in COCO. We surface the model's actual 'cow' "
        "class instead of relabeling it as 'cattle'; goats/buffalo may not "
        "be detected or may be misclassified (e.g. as 'sheep' or 'dog')."
    ),
}

DEFAULT_MODEL_WEIGHTS = "yolo11n.pt"
DEFAULT_TRACKER_CONFIG = "bytetrack.yaml"

BBox = Sequence[float]  # [x1, y1, x2, y2]


class ObjectTracker:
    """Wraps YOLO11n detection + ByteTrack identity tracking behind a single,
    stable method: `detect_and_track(frame) -> list[dict]`.

    The rest of Dharma should never need to import ultralytics, torch, or
    know anything about how detection/tracking is implemented internally.
    """

    def __init__(
        self,
        model_weights: str = DEFAULT_MODEL_WEIGHTS,
        tracker_config: str = DEFAULT_TRACKER_CONFIG,
        classes_of_interest: Optional[Iterable[str]] = COCO_CLASSES_OF_INTEREST,
        confidence_threshold: float = 0.35,
        device: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            model_weights: path or name of pretrained YOLO weights
                (default: yolo11n.pt, auto-downloaded by ultralytics on
                first use).
            tracker_config: ultralytics tracker yaml. Default is the
                built-in ByteTrack config shipped with ultralytics
                (ultralytics/cfg/trackers/bytetrack.yaml). We do not
                re-implement ByteTrack ourselves -- we use ultralytics'
                well-tested integration, consistent with the "don't train
                / don't hand-roll infra from scratch" constraint.
            classes_of_interest: iterable of COCO class *names* to keep.
                Pass None to keep all 80 COCO classes.
            confidence_threshold: minimum detection confidence to keep.
            device: "cpu", "cuda", "cuda:0", etc. Auto-detected if None.
            verbose: pass through to ultralytics (progress bars / logging).
        """
        from ultralytics import YOLO  # local import: keep this heavy dep
                                       # out of anything that just wants the
                                       # data structures / constants above.
        import torch

        self._device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = YOLO(model_weights)
        self._tracker_config = tracker_config
        self._confidence_threshold = confidence_threshold
        self._verbose = verbose

        self._class_ids_of_interest: Optional[List[int]] = None
        if classes_of_interest is not None:
            wanted = set(classes_of_interest)
            name_to_id = {name: idx for idx, name in self._model.names.items()}
            unknown = wanted - set(name_to_id)
            if unknown:
                logger.warning(
                    "classes_of_interest contains names the model does not "
                    "support and will be ignored: %s", sorted(unknown)
                )
            self._class_ids_of_interest = sorted(
                name_to_id[name] for name in wanted if name in name_to_id
            )

        logger.info(
            "ObjectTracker ready | weights=%s tracker=%s device=%s "
            "conf_thresh=%.2f classes=%s",
            model_weights, tracker_config, self._device, confidence_threshold,
            "ALL" if self._class_ids_of_interest is None
            else [self._model.names[i] for i in self._class_ids_of_interest],
        )

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------
    def detect_and_track(self, frame: np.ndarray) -> List[dict]:
        """Run detection + tracking on a single frame.

        Args:
            frame: a single BGR image as a numpy array (e.g. from
                cv2.VideoCapture.read() or cv2.imread()). Shape (H, W, 3).

        Returns:
            List of structured object dicts (see module docstring for the
            exact schema). Empty list if nothing was detected.
        """
        if frame is None:
            raise ValueError("frame is None -- nothing to detect on.")

        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self._tracker_config,
            conf=self._confidence_threshold,
            classes=self._class_ids_of_interest,
            device=self._device,
            verbose=self._verbose,
        )
        return self._to_structured_output(results[0])

    def reset(self) -> None:
        """Clear tracker state (e.g. when starting a new, unrelated video
        so old track IDs don't leak into a new sequence)."""
        # ultralytics keeps tracker state on the predictor; the officially
        # supported way to reset it is to drop the cached predictor so the
        # next .track() call rebuilds a fresh tracker.
        self._model.predictor = None

    def process_video(
        self, video_path: Union[str, Path], reset_on_start: bool = True
    ) -> Iterator[List[dict]]:
        """Convenience generator: yields structured detections frame-by-frame
        for an entire video file.

        Args:
            video_path: path to a video file readable by OpenCV.
            reset_on_start: clear any previous tracker state first, so track
                IDs from a prior video/session don't bleed into this one.

        Yields:
            One structured object list (see `detect_and_track`) per frame,
            in frame order.
        """
        import cv2

        if reset_on_start:
            self.reset()

        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield self.detect_and_track(frame)
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _to_structured_output(self, result) -> List[dict]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        track_ids = (
            boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        )

        names = self._model.names
        objects: List[dict] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            objects.append(
                {
                    "id": int(track_ids[i]) if track_ids is not None else None,
                    "class": names[cls_ids[i]],
                    "confidence": round(float(confs[i]), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "center": [round(cx, 1), round(cy, 1)],
                }
            )
        return objects


# ---------------------------------------------------------------------------
# Module-level convenience wrapper matching the spec's `detect_and_track(frame)`
# ---------------------------------------------------------------------------
_default_tracker: Optional[ObjectTracker] = None


def _get_default_tracker() -> ObjectTracker:
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ObjectTracker()
    return _default_tracker


def detect_and_track(frame: np.ndarray) -> List[dict]:
    """Module-level convenience function using a lazily-initialized default
    ObjectTracker (singleton). This matches the calling convention sketched
    in the project spec:

        from perception.detector import detect_and_track
        objects = detect_and_track(frame)

    For processing a whole video, prefer instantiating your own
    `ObjectTracker` and calling `.process_video(path)` or looping
    `.detect_and_track(frame)` yourself, so tracker state is scoped
    correctly and not shared across unrelated frame sources.
    """
    return _get_default_tracker().detect_and_track(frame)
