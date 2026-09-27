"""
Unit tests for safety/risk_engine.py (Person 4 module).

Run with:
    python -m unittest tests.test_risk_engine -v
or, if pytest is installed:
    pytest tests/test_risk_engine.py -v
"""

import sys
import os
import unittest

# Allow running this file directly (python tests/test_risk_engine.py)
# as well as via `python -m unittest` from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safety.risk_engine import RiskEngine  # noqa: E402


def make_object(obj_id, cx, cy, cls="car", confidence=0.9):
    """Helper: build a Person-1-shaped object dict."""
    return {
        "id": obj_id,
        "class": cls,
        "confidence": confidence,
        "bbox": [cx - 20, cy - 20, cx + 20, cy + 20],
        "center": [cx, cy],
    }


class TestPositionHistory(unittest.TestCase):
    """Test 1: object position history."""

    def test_history_accumulates_and_is_bounded(self):
        engine = RiskEngine(history_size=3)
        for i, (x, y) in enumerate([(100, 100), (110, 100), (120, 100), (130, 100)]):
            engine.calculate_risk([make_object(1, x, y)], timestamp=float(i))

        track = engine._tracks[1]
        # maxlen=3 -> only the last 3 positions should remain
        self.assertEqual(len(track.positions), 3)
        self.assertEqual(track.positions[-1][0], (130, 100))


class TestVelocityEstimation(unittest.TestCase):
    """Test 2: velocity estimation."""

    def test_velocity_pixels_per_second(self):
        engine = RiskEngine()
        engine.calculate_risk([make_object(1, 100, 100)], timestamp=0.0)
        # Object moved 10px in x over 1 second.
        vel = engine._update_history_and_get_velocity(1, (110.0, 100.0), 1.0)
        self.assertIsNotNone(vel)
        self.assertAlmostEqual(vel[0], 10.0, places=4)
        self.assertAlmostEqual(vel[1], 0.0, places=4)

    def test_first_frame_has_no_velocity(self):
        engine = RiskEngine()
        vel = engine._update_history_and_get_velocity(1, (100.0, 100.0), 0.0)
        self.assertIsNone(vel)


class TestRelativeMotion(unittest.TestCase):
    """Tests 3 & 4: object approaching vs moving away vs stationary."""

    def test_object_moving_toward_reference(self):
        engine = RiskEngine(stationary_speed_threshold=1.0)
        reference = (0.0, 0.0)
        # Moving from (100,0) toward (0,0): velocity points at reference.
        motion = engine._classify_relative_motion((-10.0, 0.0), (100.0, 0.0), reference)
        self.assertEqual(motion, "approaching")

    def test_object_moving_away_from_reference(self):
        engine = RiskEngine(stationary_speed_threshold=1.0)
        reference = (0.0, 0.0)
        motion = engine._classify_relative_motion((10.0, 0.0), (100.0, 0.0), reference)
        self.assertEqual(motion, "moving_away")

    def test_object_stationary(self):
        engine = RiskEngine(stationary_speed_threshold=5.0)
        reference = (0.0, 0.0)
        motion = engine._classify_relative_motion((0.5, 0.5), (100.0, 0.0), reference)
        self.assertEqual(motion, "stationary")


class TestTTC(unittest.TestCase):
    """Tests 5 & 6: TTC with positive/non-positive closing speed."""

    def test_ttc_with_positive_closing_speed(self):
        engine = RiskEngine()
        ttc = engine._time_to_collision(distance=100.0, closing_speed=50.0)
        self.assertAlmostEqual(ttc, 2.0, places=4)

    def test_ttc_is_none_when_not_closing(self):
        engine = RiskEngine()
        self.assertIsNone(engine._time_to_collision(distance=100.0, closing_speed=0.0))
        self.assertIsNone(engine._time_to_collision(distance=100.0, closing_speed=-5.0))
        self.assertIsNone(engine._time_to_collision(distance=None, closing_speed=10.0))
        self.assertIsNone(engine._time_to_collision(distance=100.0, closing_speed=None))


class TestPathConflict(unittest.TestCase):
    """Tests 7 & 8: path conflict detected / not detected (now time-stamped)."""

    def setUp(self):
        self.planned_path = [[100, 300], [200, 300], [300, 300], [400, 300]]

    def test_path_conflict_detected(self):
        engine = RiskEngine(conflict_distance=30)
        trajectory = [((250.0, 305.0), 0.0)]  # 5px from the path -> within threshold
        conflict, ttc = engine._check_path_conflict(trajectory, self.planned_path)
        self.assertTrue(conflict)
        self.assertEqual(ttc, 0.0)

    def test_no_path_conflict(self):
        engine = RiskEngine(conflict_distance=30)
        trajectory = [((250.0, 500.0), 0.0)]  # 200px away -> outside threshold
        conflict, ttc = engine._check_path_conflict(trajectory, self.planned_path)
        self.assertFalse(conflict)
        self.assertIsNone(ttc)

    def test_no_path_conflict_when_path_missing(self):
        engine = RiskEngine()
        trajectory = [((250.0, 300.0), 0.0)]
        conflict, ttc = engine._check_path_conflict(trajectory, None)
        self.assertFalse(conflict)
        self.assertIsNone(ttc)
        conflict2, ttc2 = engine._check_path_conflict(trajectory, [])
        self.assertFalse(conflict2)
        self.assertIsNone(ttc2)

    def test_earliest_future_conflict_time_is_returned(self):
        # Object currently far from the path, but its projected future
        # trajectory crosses it -- this is the anticipatory behaviour.
        engine = RiskEngine(conflict_distance=20)
        trajectory = [
            ((250.0, 400.0), 0.0),   # far now
            ((250.0, 350.0), 1.0),   # still far
            ((250.0, 305.0), 2.0),   # within threshold -> earliest conflict
            ((250.0, 300.0), 3.0),   # also within threshold, but later
        ]
        conflict, ttc = engine._check_path_conflict(trajectory, self.planned_path)
        self.assertTrue(conflict)
        self.assertAlmostEqual(ttc, 2.0, places=4)


class TestRiskClassification(unittest.TestCase):
    """Tests 9, 10, 11: HIGH_RISK / SAFE / CAUTION classification."""

    def test_high_risk_classification(self):
        engine = RiskEngine(high_risk_ttc=2.0, caution_ttc=4.0)
        risk = engine._classify_risk(
            ttc=1.0,
            path_conflict=True,
            distance=20.0,
            relative_motion="approaching",
            object_class="person",
            ego_speed=0.0,
        )
        self.assertEqual(risk, "HIGH_RISK")

    def test_safe_classification_moving_away(self):
        engine = RiskEngine()
        risk = engine._classify_risk(
            ttc=None,
            path_conflict=False,
            distance=500.0,
            relative_motion="moving_away",
            object_class="car",
            ego_speed=0.0,
        )
        self.assertEqual(risk, "SAFE")

    def test_caution_classification(self):
        engine = RiskEngine(high_risk_ttc=1.0, caution_ttc=5.0)
        risk = engine._classify_risk(
            ttc=3.0,
            path_conflict=True,
            distance=100.0,
            relative_motion="approaching",
            object_class="car",
            ego_speed=0.0,
        )
        self.assertEqual(risk, "CAUTION")

    def test_vulnerable_class_is_more_conservative(self):
        # Same TTC, same conditions, but a "person" should escalate to
        # CAUTION sooner than a "car" because of the priority multiplier
        # applied to the anticipatory early-warning window (safe_ttc).
        engine = RiskEngine(high_risk_ttc=1.0, safe_ttc=2.0)
        car_risk = engine._classify_risk(
            ttc=2.5,
            path_conflict=True,
            distance=100.0,
            relative_motion="approaching",
            object_class="car",
            ego_speed=0.0,
        )
        person_risk = engine._classify_risk(
            ttc=2.5,
            path_conflict=True,
            distance=100.0,
            relative_motion="approaching",
            object_class="person",
            ego_speed=0.0,
        )
        self.assertEqual(car_risk, "SAFE")
        self.assertEqual(person_risk, "CAUTION")


class TestRobustness(unittest.TestCase):
    """Test 12: malformed / empty input must not crash the pipeline."""

    def test_empty_object_list(self):
        engine = RiskEngine()
        self.assertEqual(engine.calculate_risk([]), [])
        self.assertEqual(engine.calculate_risk(None), [])

    def test_object_with_id_none(self):
        engine = RiskEngine()
        obj = make_object(None, 100, 100)
        results = engine.calculate_risk([obj])
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0]["object_id"])
        self.assertEqual(results[0]["risk"], "SAFE")

    def test_missing_center_is_skipped(self):
        engine = RiskEngine()
        bad_obj = {"id": 1, "class": "car", "confidence": 0.9, "bbox": [0, 0, 10, 10]}
        results = engine.calculate_risk([bad_obj])
        self.assertEqual(results, [])

    def test_malformed_bbox_does_not_crash(self):
        engine = RiskEngine()
        obj = make_object(1, 100, 100)
        obj["bbox"] = "not-a-bbox"
        results = engine.calculate_risk([obj])
        self.assertEqual(len(results), 1)  # bbox isn't used for math, so it's fine

    def test_non_dict_object_in_list_is_skipped(self):
        engine = RiskEngine()
        results = engine.calculate_risk([make_object(1, 100, 100), "garbage", None, 42])
        self.assertEqual(len(results), 1)

    def test_zero_or_negative_dt_does_not_crash(self):
        engine = RiskEngine()
        engine.calculate_risk([make_object(1, 100, 100)], timestamp=5.0)
        # Same timestamp again (dt == 0)
        results = engine.calculate_risk([make_object(1, 105, 100)], timestamp=5.0)
        self.assertEqual(results[0]["ttc"], None)
        # Earlier timestamp (dt < 0)
        results2 = engine.calculate_risk([make_object(1, 110, 100)], timestamp=4.0)
        self.assertEqual(results2[0]["ttc"], None)

    def test_missing_prediction_falls_back_to_extrapolation(self):
        engine = RiskEngine()
        engine.calculate_risk([make_object(1, 100, 300)], timestamp=0.0)
        results = engine.calculate_risk(
            [make_object(1, 110, 300)],
            predictions=None,
            planned_path=[[100, 300], [400, 300]],
            timestamp=1.0,
        )
        self.assertEqual(len(results), 1)  # must not crash without predictions

    def test_missing_and_empty_planned_path(self):
        engine = RiskEngine()
        obj = make_object(1, 100, 100)
        r1 = engine.calculate_risk([obj], planned_path=None, timestamp=0.0)
        r2 = engine.calculate_risk([obj], planned_path=[], timestamp=1.0)
        self.assertFalse(r1[0]["path_conflict"])
        self.assertFalse(r2[0]["path_conflict"])

    def test_object_disappearing_and_reappearing(self):
        engine = RiskEngine()
        engine.calculate_risk([make_object(1, 100, 100)], timestamp=0.0)
        # Object 1 vanishes for a few frames (Person 1 stops reporting it).
        engine.calculate_risk([make_object(2, 400, 400)], timestamp=1.0)
        engine.calculate_risk([make_object(2, 410, 400)], timestamp=2.0)
        # Object 1 reappears -- should not crash, history simply continues.
        results = engine.calculate_risk([make_object(1, 105, 100)], timestamp=3.0)
        self.assertEqual(len(results), 1)


class TestAnticipatoryEarlyWarning(unittest.TestCase):
    """
    Proves the anticipation fix: a fast object that is still FAR from the
    ego path must be flagged before it becomes geometrically close, based
    on its projected trajectory -- not only once distance/closing-speed
    say it's already near.
    """

    def test_fast_far_object_is_flagged_before_it_gets_close(self):
        engine = RiskEngine(
            base_distance=60,
            conflict_distance=50,
            safe_ttc=6.0,
            high_risk_ttc=2.5,
            prediction_horizon_seconds=4.0,
            prediction_step_seconds=0.5,
        )
        planned_path = [[0, 300], [200, 300], [400, 300], [600, 300]]

        # A vehicle moving fast (200 px/s) toward the path, but still far
        # away right now (700 px from the nearest path point).
        engine.calculate_risk(
            [make_object(1, 300, 1000, cls="car")],
            planned_path=planned_path,
            timestamp=0.0,
        )
        results = engine.calculate_risk(
            [make_object(1, 300, 900, cls="car")],  # moved 100px in 0.5s -> 200 px/s
            planned_path=planned_path,
            timestamp=0.5,
        )
        result = results[0]

        # It is nowhere near the path right now (distance is large)...
        self.assertGreater(result["distance"], 500)
        # ...but the projected trajectory shows it will reach the path
        # within the prediction horizon, so it must already be flagged.
        self.assertIsNotNone(result["predicted_conflict_ttc"])
        self.assertIn(result["risk"], ("CAUTION", "HIGH_RISK"))


class TestSyntheticDemo(unittest.TestCase):
    """Deterministic end-to-end scenario, mirroring the synthetic demo."""

    def test_object_on_collision_path_is_flagged(self):
        engine = RiskEngine(
            history_size=5,
            conflict_distance=20,
            high_risk_ttc=2.0,
            caution_ttc=5.0,
        )
        planned_path = [[250, 300], [300, 300], [350, 300], [400, 300]]

        engine.calculate_risk(
            [make_object(1, 300, 300)], planned_path=planned_path, timestamp=0.0
        )
        engine.calculate_risk(
            [make_object(1, 310, 300)], planned_path=planned_path, timestamp=1.0
        )
        results = engine.calculate_risk(
            [make_object(1, 320, 300)], planned_path=planned_path, timestamp=2.0
        )

        result = results[0]
        self.assertTrue(result["path_conflict"])
        self.assertIn(result["risk"], ("CAUTION", "HIGH_RISK"))


if __name__ == "__main__":
    unittest.main()