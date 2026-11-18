import unittest
from unittest.mock import patch
from sweep_defensive_strength import STRENGTHS, run_game, make_summary
from monte_carlo.heuristic import DEFAULT_HEURISTIC

class DefensiveSweepTests(unittest.TestCase):
    def test_requested_range_and_zero_control(self):
        self.assertEqual(STRENGTHS, (0, .05, .075, .10, .15, .20, .30, .50, 1))

    def test_worker_changes_only_strength(self):
        for strength in STRENGTHS:
            with patch("sweep_defensive_strength.play_game", return_value={}) as play:
                result = run_game((strength, 1281000, 1, {}))
            self.assertEqual(play.call_args.args,
                (1, 1281000, 20, 4, 10000, "baseline", "heuristic",
                 DEFAULT_HEURISTIC, "defensive", strength))
            self.assertEqual(result["strength"], strength)

    def test_summary_tracks_all_900_games(self):
        result = make_summary([], {"strengths": STRENGTHS,
            "seeds": [1281000,1282000,1283000,1284000,1285000], "games_per_batch":20})
        self.assertEqual(result["expected_games"], 900)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["candidates"], [])

if __name__ == "__main__":
    unittest.main()
