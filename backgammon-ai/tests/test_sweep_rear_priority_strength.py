import unittest
from unittest.mock import patch
from sweep_rear_priority_strength import STRENGTHS, run_game, make_summary
from monte_carlo.heuristic import DEFAULT_HEURISTIC

class RearPrioritySweepTests(unittest.TestCase):
    def test_requested_range_and_zero_control(self):
        self.assertEqual(STRENGTHS, (0, .05, .075, .10, .15, .20, .30, .50, 1))

    def test_worker_changes_only_strength(self):
        for strength in STRENGTHS:
            with patch("sweep_rear_priority_strength.play_game", return_value={}) as play:
                result = run_game((strength, 1181000, 1, {}))
            self.assertEqual(play.call_args.args,
                (1, 1181000, 20, 4, 10000, "baseline", "heuristic",
                 DEFAULT_HEURISTIC, "rear-priority", strength))
            self.assertEqual(result["strength"], strength)

    def test_summary_tracks_all_900_games(self):
        result = make_summary([], {"strengths": STRENGTHS,
            "seeds": [1181000,1182000,1183000,1184000,1185000], "games_per_batch":20})
        self.assertEqual(result["expected_games"], 900)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["candidates"], [])

if __name__ == "__main__":
    unittest.main()
