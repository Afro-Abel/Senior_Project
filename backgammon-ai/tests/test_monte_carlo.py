import random
import unittest
from unittest.mock import patch

from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.move import Step, OFF
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.rollout import rollout


def race(player=1):
    state = GameState()
    state.board[2] = 1
    state.board[21] = -1
    state.borne_off = {1: 14, -1: 14}
    state.current_player = player
    return state


class MonteCarloTests(unittest.TestCase):
    def test_budget_must_be_positive_integer(self):
        for budget in (0, -1, 1.5, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                MonteCarloAgent(budget)

    def test_budget_perspective_next_player_and_state_preservation(self):
        for player in (1, -1):
            state = race(player)
            state.board[2] = state.board[21] = 0
            state.board[5] = 1
            state.board[18] = -1
            snapshot = state.copy()
            seen = []

            def rollout(after, rng, depth, perspective, policy, config):
                self.assertEqual(policy, "random")
                self.assertEqual(depth, 4)
                self.assertEqual(perspective, player)
                self.assertEqual(after.current_player, -player)
                for p in (1, -1):
                    self.assertEqual(after.checker_count(p), 15)
                seen.append(after.copy())
                return 0.6 if len(seen) % 3 else -0.3

            with patch("agents.monte_carlo_agent.rollout", side_effect=rollout):
                results = MonteCarloAgent(3).evaluate_moves(state, (1, 2))
            self.assertEqual(state, snapshot)
            self.assertTrue(all(e.move in get_legal_moves(state, (1, 2), player)
                                for e in results))
            nonterminal = [e for e in results if not e.immediate_win]
            self.assertTrue(nonterminal)
            for e in nonterminal:
                self.assertAlmostEqual(e.score, 0.3)
                self.assertEqual(e.rollouts, 3)
            self.assertEqual(len(seen), 3 * len(nonterminal))

    def test_selects_highest_mean_utility(self):
        state = race()
        state.board[2], state.board[5] = 0, 2
        state.borne_off[1] = 13
        with patch("agents.monte_carlo_agent.rollout",
                   side_effect=lambda after, *args: 0.8 if after.board[2] else -0.2):
            chosen = MonteCarloAgent(3).choose_move(state, (1, 2))
        for step in chosen.steps:
            state = apply_step(state, step, 1)
        self.assertEqual(state.board[2], 1)

    def test_opening_deduplicates_positions_and_simulation_budget(self):
        state = GameState.initial_state()
        state.current_player = -1
        snapshot = state.copy()
        legal = get_legal_moves(state, (2, 6), -1)
        self.assertEqual(len(legal), 28)

        def result(move):
            after = state.copy()
            for step in move.steps:
                after = apply_step(after, step, -1)
            return (tuple(after.board), tuple(after.bar[p] for p in (1, -1)),
                    tuple(after.borne_off[p] for p in (1, -1)))

        representatives = {}
        for move in legal:
            representatives.setdefault(result(move), move)
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as rollout:
            evaluations = MonteCarloAgent(3).evaluate_moves(state, (2, 6))
        self.assertEqual(len(evaluations), 14)
        self.assertEqual([e.move for e in evaluations], list(representatives.values()))
        self.assertEqual({result(e.move) for e in evaluations}, set(representatives))
        self.assertEqual(rollout.call_count, 14 * 3)
        self.assertEqual(state, snapshot)

    def test_proven_win_beats_noisy_perfect_estimate(self):
        state = race()
        state.board[0] = 1
        state.borne_off[1] = 13
        with patch("agents.monte_carlo_agent.rollout", return_value=1):
            move = MonteCarloAgent(1).choose_move(state, (3, 1))
        self.assertEqual({s.end for s in move.steps}, {OFF})

    def test_real_rollouts_and_legal_reproducible_choice(self):
        state = race()
        state.board[2] = 0
        state.board[5] = 1
        snapshot = state.copy()
        first = MonteCarloAgent(3, random.Random(9))
        second = MonteCarloAgent(3, random.Random(9))
        chosen = first.choose_move(state, (1, 2))
        self.assertEqual(chosen, second.choose_move(state, (1, 2)))
        self.assertIn(chosen, get_legal_moves(state, (1, 2), 1))
        self.assertEqual(state, snapshot)

    def test_immediate_win_needs_no_rollouts_for_either_player(self):
        for player in (1, -1):
            state = race(player)
            with patch("agents.monte_carlo_agent.rollout") as rollout:
                move = MonteCarloAgent(3).choose_move(state, (3, 6))
            self.assertEqual(move.steps, [Step(3 if player == 1 else 22, OFF)])
            rollout.assert_not_called()

    def test_forced_pass_needs_no_rollouts(self):
        state = GameState()
        state.bar[1] = 15
        state.board[23] = -2
        state.board[22] = -2
        state.borne_off[-1] = 11
        with patch("agents.monte_carlo_agent.rollout") as rollout:
            self.assertIsNone(MonteCarloAgent().choose_move(state, (1, 2)))
        rollout.assert_not_called()

    def test_finished_game_is_rejected(self):
        state = race()
        state.board[2] = 0
        state.borne_off[1] = 15
        with self.assertRaises(ValueError):
            MonteCarloAgent().choose_move(state, (1, 2))
        with patch("backgammon.game.BackgammonGame.roll_dice") as dice:
            self.assertEqual(rollout(state, random.Random(1), 4, 1), 1)
        dice.assert_not_called()

    def test_rollout_passes_blocked_player_then_reaches_winner(self):
        state = GameState()
        state.bar[1] = 1
        state.borne_off = {1: 14, -1: 11}
        state.board[23] = -2
        state.board[22] = -2
        snapshot = state.copy()
        with patch("backgammon.game.BackgammonGame.roll_dice",
                   side_effect=[(1, 2), (6, 6)]) as dice:
            self.assertEqual(rollout(state, random.Random(2), 4, 1), -1)
        self.assertEqual(dice.call_count, 2)
        self.assertEqual(state, snapshot)


if __name__ == "__main__":
    unittest.main()
