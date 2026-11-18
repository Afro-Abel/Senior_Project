import random
import unittest
from unittest.mock import patch

from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import evaluate_position
from monte_carlo.rollout import choose_heuristic_move, rollout


class RolloutPolicyTests(unittest.TestCase):
    def test_turn_order_and_cutoff_perspective_for_both_policies(self):
        for policy in ("random", "heuristic"):
            state = GameState.initial_state()
            state.current_player = -1  # Opponent moves after original P1 candidate.
            snapshot = state.copy()
            turns = []

            def legal(current, dice, player):
                turns.append(player)
                self.assertEqual(current.current_player, player)
                return get_legal_moves(current, dice, player)

            with patch("monte_carlo.rollout.get_legal_moves", side_effect=legal), \
                 patch("monte_carlo.rollout.evaluate_position", wraps=evaluate_position) as score:
                result = rollout(state, random.Random(3), 4, 1, policy)
            self.assertEqual(turns, [-1, 1, -1, 1])
            # Internal heuristic choices may use either perspective. Final score
            # must always use original decision maker, even on opponent's turn.
            cutoff, perspective = score.call_args.args[:2]
            self.assertEqual(perspective, 1)
            self.assertEqual(result, evaluate_position(cutoff, 1))
            self.assertEqual(state, snapshot)

    def test_guided_policy_chooses_best_for_acting_player(self):
        for player in (1, -1):
            state = GameState.initial_state()
            state.current_player = player
            snapshot = state.copy()
            moves = get_legal_moves(state, (2, 6), player)
            values = []
            for move in moves:
                after = state.copy()
                for step in move.steps:
                    after = apply_step(after, step, player)
                values.append(evaluate_position(after, player))
            chosen = choose_heuristic_move(state, moves, random.Random(1))
            self.assertIn(chosen, moves)
            self.assertEqual(values[moves.index(chosen)], max(values))
            self.assertEqual(state, snapshot)

    def test_guided_policy_deduplicates_before_scoring(self):
        state = GameState.initial_state()
        state.current_player = -1
        moves = get_legal_moves(state, (2, 6), -1)
        with patch("monte_carlo.rollout.evaluate_position", return_value=0) as score:
            self.assertIn(choose_heuristic_move(state, moves, random.Random(1)), moves)
        self.assertEqual(score.call_count, 14)

    def test_guided_decision_is_seeded_legal_and_preserves_state(self):
        state = GameState()
        state.board[5], state.board[18] = 2, -2
        state.borne_off = {1: 13, -1: 13}
        snapshot = state.copy()
        first = MonteCarloAgent(2, random.Random(9), depth=4, rollout_policy="heuristic")
        second = MonteCarloAgent(2, random.Random(9), depth=4, rollout_policy="heuristic")
        move = first.choose_move(state, (1, 2))
        self.assertEqual(move, second.choose_move(state, (1, 2)))
        self.assertIn(move, get_legal_moves(state, (1, 2), 1))
        self.assertEqual(state, snapshot)

    def test_guided_pass_and_terminal_stop(self):
        state = GameState()
        state.bar[1] = 1
        state.borne_off = {1: 14, -1: 11}
        state.board[23], state.board[22] = -2, -2
        with patch("backgammon.game.BackgammonGame.roll_dice",
                   side_effect=[(1, 2), (6, 6)]) as dice:
            self.assertEqual(rollout(state, random.Random(1), 4, 1, "heuristic"), -1)
        self.assertEqual(dice.call_count, 2)

    def test_invalid_policy(self):
        with self.assertRaises(ValueError):
            MonteCarloAgent(rollout_policy="unknown")
        with self.assertRaises(ValueError):
            rollout(GameState.initial_state(), random.Random(1), 1, 1, "unknown")


if __name__ == "__main__":
    unittest.main()
