import unittest
from unittest.mock import patch

from compare_styles import AGENTS, make_summary, run_game
from monte_carlo.heuristic import DEFAULT_HEURISTIC


class CompareStyleTests(unittest.TestCase):
    def test_worker_holds_search_and_opponent_fixed(self):
        self.assertEqual(len(AGENTS), 5)
        for agent in AGENTS:
            with patch("compare_styles.play_game", return_value={}) as play:
                result = run_game((agent, 1121000, 1, {"style_strength": .05}))
            self.assertEqual(play.call_args.args[:7], (1, 1121000, 20, 4, 10000, "baseline", "heuristic"))
            self.assertEqual(play.call_args.args[7:], (DEFAULT_HEURISTIC, agent, .05))
            self.assertEqual(result["agent"], agent)

    def test_partial_summary_does_not_claim_completion(self):
        summary = make_summary([], {"seeds": [1121000, 1122000], "games_per_batch": 20})
        self.assertEqual(summary["expected_games"], 200)
        self.assertEqual(summary["completed_games"], 0)
        self.assertEqual(summary["status"], "running")


if __name__ == "__main__":
    unittest.main()
