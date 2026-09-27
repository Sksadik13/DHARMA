"""
DHARMA - safety/demo_synthetic.py

Standalone, deterministic demonstration of RiskEngine that does NOT require
YOLO, a camera, or any of the other team modules. Run this directly to see
the risk engine work end-to-end on hand-crafted data.

Windows CMD:
    D:\\Dharma2> .venv\\Scripts\\activate
    (.venv) D:\\Dharma2> python safety\\demo_synthetic.py
"""

from safety.risk_engine import RiskEngine


def make_object(obj_id, cx, cy, cls="car"):
    return {
        "id": obj_id,
        "class": cls,
        "confidence": 0.9,
        "bbox": [cx - 20, cy - 20, cx + 20, cy + 20],
        "center": [cx, cy],
    }


def main() -> None:
    engine = RiskEngine(
        history_size=5,
        conflict_distance=25,
        base_distance=40,
        safe_ttc=5.0,
        caution_ttc=3.0,
        high_risk_ttc=1.5,
    )

    # A vehicle path straight along y = 300.
    planned_path = [[250, 300], [300, 300], [350, 300], [400, 300]]

    # Object 1: approaches the planned path head-on, frame by frame.
    frames = [
        # (timestamp, center)
        (0.0, (300, 250)),
        (1.0, (300, 270)),
        (2.0, (300, 288)),
    ]

    print("=== DHARMA synthetic safety demo ===")
    print(f"Planned path: {planned_path}\n")

    for t, center in frames:
        objects = [make_object(1, center[0], center[1], cls="person")]
        results = engine.calculate_risk(
            objects=objects,
            predictions=None,       # Person 3 not wired up yet
            road=None,               # Person 2 not wired up yet
            planned_path=planned_path,
            timestamp=t,
        )
        print(f"t={t:>4.1f}  object center={center}  ->  {results[0]}")

    print("\nDone. See README / risk_engine.py docstring for field meanings.")


if __name__ == "__main__":
    main()