import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sweep_borne_off import SEEDS, WEIGHTS, atomic_json, results_summary, run_game


class BorneOffSweepTests(unittest.TestCase):
    def test_design_has_100_balanced_games_per_weight(self):
        self.assertEqual(WEIGHTS, (0, 2, 4, 6, 8))
        self.assertEqual(len(SEEDS) * 20, 100)
        streams = [set(range(seed, seed + 60)) for seed in SEEDS]
        for i, left in enumerate(streams):
            for right in streams[i + 1:]:
                self.assertFalse(left & right)
        self.assertNotIn(1000000, set.union(*streams))

    def test_worker_changes_only_candidate_borne_off_weight(self):
        settings = {"rollouts": 20, "depth": 4, "max_turns": 10000}
        with patch("sweep_borne_off.play_game", return_value={"mc_won": True}) as play:
            result = run_game((6, 980100, 0, settings))
        args = play.call_args.args
        self.assertEqual(args[:7], (0, 980100, 20, 4, 10000, "baseline", "heuristic"))
        self.assertEqual(args[-1].borne_off, 6)
        self.assertEqual(args[-1].bar, 8)
        self.assertEqual(result["heuristic_config"]["scale"], 50)

    def test_partial_summary_and_atomic_checkpoints(self):
        settings = {"weights": [0, 2, 4, 6, 8], "seeds": list(SEEDS), "games_per_batch": 20}
        record = {"weight": 4, "elapsed_seconds": 1.2,
                  "game": {"mc_won": True, "mc_player": 1, "starter": 1,
                           "turns": 50, "decision_seconds": [.2]}}
        summary = results_summary([record], settings)
        self.assertEqual(summary["status"], "running")
        self.assertEqual(summary["completed_games"], 1)
        self.assertEqual(summary["expected_games"], 500)
        self.assertEqual(summary["candidates"][0]["borne_off"], 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            atomic_json(path, summary)
            self.assertEqual(json.loads(path.read_text()), summary)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
