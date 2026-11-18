import unittest
from unittest.mock import patch

from backgammon.rules import get_legal_moves
from evaluate_baseline import play_game, summarize, wilson_interval
from monte_carlo.heuristic import DEFAULT_HEURISTIC
from dataclasses import replace


def cheap_move(agent, state, dice):
    moves = get_legal_moves(state, dice, state.current_player)
    return agent.rng.choice(moves) if moves else None


class EvaluationTests(unittest.TestCase):
    def test_summary_and_interval(self):
        records = [
            {"mc_won": True, "mc_player": 1, "starter": 1,
             "turns": 20, "decision_seconds": [.1, .3]},
            {"mc_won": False, "mc_player": -1, "starter": 1,
             "turns": 40, "decision_seconds": [.2]},
        ]
        result = summarize(records)
        self.assertEqual(result["mc_win_rate"], .5)
        self.assertEqual(result["opponent"], "random")
        self.assertEqual(result["opponent_wins"], 1)
        heuristic_result = summarize(records, "heuristic")
        self.assertEqual(heuristic_result["opponent"], "heuristic")
        self.assertNotIn("random_wins", heuristic_result)
        self.assertEqual(result["mean_turns"], 30)
        self.assertAlmostEqual(result["mean_decision_seconds"], .2)
        self.assertEqual(result["p95_decision_seconds"], .3)
        self.assertEqual(result["by_mc_player"]["1"], {"games": 1, "wins": 1})
        low, high = wilson_interval(1, 2)
        self.assertLess(low, .5)
        self.assertGreater(high, .5)
        self.assertAlmostEqual(low, 1 - high)

    def test_complete_games_balanced_seats_and_reproducible_outcomes(self):
        # Search is tested elsewhere; cheaply exercise the complete runner here.
        with patch("evaluate_baseline.MonteCarloAgent.choose_move", cheap_move):
            first = play_game(0, 700, 1, 1, 10000)
            replay = play_game(0, 700, 1, 1, 10000)
            second = play_game(1, 700, 1, 1, 10000)
        self.assertEqual(first["mc_player"], 1)
        self.assertEqual(second["mc_player"], -1)
        self.assertTrue(first["decision_seconds"])
        self.assertNotEqual(first["dice_seed"], second["dice_seed"])
        first.pop("decision_seconds")
        replay.pop("decision_seconds")
        self.assertEqual(first, replay)

    def test_turn_limit_raises_instead_of_recording_draw(self):
        with patch("evaluate_baseline.MonteCarloAgent.choose_move", cheap_move):
            with self.assertRaisesRegex(RuntimeError, "exceeded 1 turns"):
                play_game(0, 700, 1, 1, 1)

    def test_heuristic_opponent_uses_depth_zero_for_both_seats(self):
        calls = []

        def tracked_move(agent, state, dice):
            calls.append((state.current_player, agent.depth, agent.rollouts_per_move))
            return cheap_move(agent, state, dice)

        with patch("evaluate_baseline.MonteCarloAgent.choose_move", tracked_move):
            for index in (0, 1):
                calls.clear()
                result = play_game(index, 700, 3, 4, 10000, "heuristic")
                self.assertEqual(result["opponent"], "heuristic")
                self.assertNotIn("random_seed", result)
                self.assertTrue(any(p == result["mc_player"] and d == 4 and r == 3
                                    for p, d, r in calls))
                self.assertTrue(any(p == -result["mc_player"] and d == 0 and r == 1
                                    for p, d, r in calls))

    def test_invalid_opponent_is_rejected(self):
        with self.assertRaises(ValueError):
            play_game(0, 700, 1, 1, 10000, "unknown")

    def test_candidate_parameters_do_not_change_frozen_opponents(self):
        config = replace(DEFAULT_HEURISTIC, borne_off=0)
        for opponent_kind in ("heuristic", "baseline"):
            for index in (0, 1):
                calls = []

                def tracked_move(agent, state, dice):
                    calls.append((state.current_player, agent.heuristic_config,
                                  agent.depth, agent.rollouts_per_move))
                    return cheap_move(agent, state, dice)

                with patch("evaluate_baseline.MonteCarloAgent.choose_move", tracked_move):
                    record = play_game(index, 700, 3, 1, 10000, opponent_kind,
                                       "heuristic", config)
                main = record["mc_player"]
                self.assertTrue(any(p == main for p, *_ in calls))
                self.assertTrue(any(p == -main for p, *_ in calls))
                for player, actual, depth, rollouts in calls:
                    if player == main:
                        self.assertEqual((actual, depth, rollouts), (config, 1, 3))
                    else:
                        expected = (4, 20) if opponent_kind == "baseline" else (0, 1)
                        self.assertEqual((actual, depth, rollouts), (DEFAULT_HEURISTIC, *expected))

    def test_default_random_policy_preserves_recorded_game(self):
        result = play_game(0, 20261017, 1, 1, 10000)
        self.assertEqual((result["winner"], result["turns"]), (1, 51))
        self.assertEqual(result["rollout_policy"], "random")

    def test_rejects_illegal_agent_output(self):
        # Pick a game index/seed whose opening player is the MC seat.
        from backgammon.move import Move, Step
        with patch("evaluate_baseline.MonteCarloAgent.choose_move",
                   return_value=Move([Step(24, 0)])):
            with self.assertRaises(RuntimeError) as caught:
                play_game(0, 700, 1, 1, 10000)
        self.assertIsInstance(caught.exception.__cause__, AssertionError)
        self.assertIn("illegal", str(caught.exception.__cause__))


if __name__ == "__main__":
    unittest.main()
