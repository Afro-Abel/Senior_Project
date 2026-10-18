import random
import unittest
from unittest.mock import patch

from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.state import GameState
from monte_carlo.heuristic import evaluate_position, pip_count
from monte_carlo.rollout import rollout


class HeuristicTests(unittest.TestCase):
    def test_initial_symmetry_and_pip_counts(self):
        state = GameState.initial_state()
        self.assertEqual(pip_count(state, 1), 167)
        self.assertEqual(pip_count(state, -1), 167)
        self.assertEqual(evaluate_position(state, 1), 0)

    def test_perspective_and_board_reflection(self):
        state = GameState.initial_state()
        state.board[23] -= 1
        state.bar[1] = 1
        reflected = GameState(
            [-x for x in reversed(state.board)],
            {p: state.bar[-p] for p in (1, -1)},
            {p: state.borne_off[-p] for p in (1, -1)}, -1)
        score = evaluate_position(state, 1)
        self.assertLess(score, 0)
        self.assertAlmostEqual(score, -evaluate_position(state, -1))
        self.assertAlmostEqual(score, evaluate_position(reflected, -1))
        self.assertLess(abs(score), 1)

    def test_race_progress_and_bearing_off_improve_score(self):
        state = GameState()
        state.board[5], state.board[18] = 1, -1
        state.borne_off = {1: 14, -1: 14}
        before = evaluate_position(state, 1)
        state.board[5], state.board[4] = 0, 1
        self.assertGreater(evaluate_position(state, 1), before)
        state.board[4] = 0
        state.borne_off[1] = 15
        self.assertEqual(evaluate_position(state, 1), 1)
        self.assertEqual(evaluate_position(state, -1), -1)

    def test_protection_improves_equal_pip_contact_position(self):
        safe = GameState()
        safe.board[7], safe.board[0] = 2, -15
        safe.borne_off[1] = 13
        exposed = safe.copy()
        exposed.board[7] = 0
        exposed.board[6] = exposed.board[8] = 1
        self.assertEqual(pip_count(safe, 1), pip_count(exposed, 1))
        self.assertGreater(evaluate_position(safe, 1), evaluate_position(exposed, 1))

    def test_pure_race_ignores_protection_at_equal_progress(self):
        safe = GameState()
        safe.board[3], safe.board[18] = 2, -15
        safe.borne_off[1] = 13
        spread = safe.copy()
        spread.board[3] = 0
        spread.board[2] = spread.board[4] = 1
        self.assertEqual(evaluate_position(safe, 1), evaluate_position(spread, 1))

    def test_depth_is_exact_turn_limit_and_preserves_input(self):
        state = GameState.initial_state()
        snapshot = state.copy()
        with patch("backgammon.game.BackgammonGame.roll_dice", return_value=(1, 2)) as dice:
            score = rollout(state, random.Random(3), 3, 1)
        self.assertEqual(dice.call_count, 3)
        self.assertTrue(-1 < score < 1)
        self.assertEqual(state, snapshot)

    def test_pass_consumes_depth(self):
        state = GameState()
        state.bar[1] = 1
        state.borne_off = {1: 14, -1: 11}
        state.board[23], state.board[22] = -2, -2
        with patch("backgammon.game.BackgammonGame.roll_dice", side_effect=[(1, 2)]) as dice:
            score = rollout(state, random.Random(1), 1, 1)
        self.assertEqual(dice.call_count, 1)
        self.assertEqual(score, evaluate_position(state, 1))
        self.assertNotEqual(score, -1)

    def test_depth_zero_does_not_roll_dice(self):
        state = GameState.initial_state()
        with patch("backgammon.game.BackgammonGame.roll_dice") as dice:
            results = MonteCarloAgent(10, depth=0).evaluate_moves(state, (1, 2))
        dice.assert_not_called()
        self.assertTrue(all(e.rollouts == 1 for e in results))

    def test_invalid_depth(self):
        for depth in (-1, 1.5, True):
            with self.subTest(depth=depth):
                with self.assertRaises(ValueError):
                    MonteCarloAgent(depth=depth)
                with self.assertRaises(ValueError):
                    rollout(GameState.initial_state(), random.Random(1), depth, 1)


if __name__ == "__main__":
    unittest.main()
