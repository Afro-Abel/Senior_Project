import unittest
from unittest.mock import patch

from confirm_borne_off import SEEDS, WEIGHTS, paired_comparison, results_summary, run_game


class ConfirmationTests(unittest.TestCase):
    def test_predefined_design_and_frozen_external_opponent(self):
        self.assertEqual(WEIGHTS, (0, 4))
        self.assertEqual(len(SEEDS) * 20, 100)
        self.assertEqual(SEEDS[0], 1000000)
        for left, right in zip(SEEDS, SEEDS[1:]):
            self.assertGreaterEqual(right - left, 60)
        with patch("confirm_borne_off.play_game", return_value={}) as play:
            result = run_game((0, SEEDS[0], 0, {"rollouts": 20, "depth": 4, "max_turns": 10000}))
        self.assertEqual(play.call_args.args[5:7], ("gnubg", "heuristic"))
        self.assertEqual(result["heuristic_config"]["borne_off"], 0)
        self.assertEqual(result["heuristic_config"]["bar"], 8)

    def test_matched_comparison_and_exact_probability(self):
        records = []
        for index in range(10):
            for weight in WEIGHTS:
                records.append({"weight": weight, "batch_seed": 1, "index": index,
                                "game": {"mc_won": weight == 0}})
        comparison = paired_comparison(records)
        self.assertEqual(comparison["weight0_only_wins"], 10)
        self.assertEqual(comparison["weight4_only_wins"], 0)
        self.assertAlmostEqual(comparison["two_sided_exact_discordance_p"], 2 / 1024)
        for record in records:
            record["game"]["mc_won"] = True
        self.assertEqual(paired_comparison(records)["two_sided_exact_discordance_p"], 1)

    def test_partial_summary_withholds_comparison(self):
        result = results_summary([], {"weights": list(WEIGHTS), "seeds": list(SEEDS), "games_per_batch": 20})
        self.assertEqual(result["expected_games"], 200)
        self.assertEqual(result["status"], "running")
        self.assertIsNone(result["matched_comparison"])


if __name__ == "__main__":
    unittest.main()
