import importlib.util
import random
import unittest
from unittest.mock import patch

from agents.gnubg_agent import GNUBGAgent, to_gnubg_board
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState


def position(points, bars=None):
    state = GameState()
    for point, count in points.items():
        state.board[point - 1] = count
    if bars:
        state.bar.update(bars)
    for p in (1, -1):
        state.borne_off[p] = 15 - state.checker_count(p)
    return state


class BoardConversionTests(unittest.TestCase):
    def test_initial_distances_and_player_rows(self):
        state = GameState.initial_state()
        expected = [0] * 25
        for index, count in ((5, 5), (7, 3), (12, 5), (23, 2)):
            expected[index] = count
        for player in (1, -1):
            self.assertEqual(to_gnubg_board(state, player), [expected, expected])

    def test_asymmetric_points_bar_and_implicit_off(self):
        state = position({3: 2, 19: -4}, {1: 1, -1: 2})
        snapshot = state.copy()
        opponent, own = to_gnubg_board(state, 1)
        self.assertEqual((own[2], own[24], 15 - sum(own)), (2, 1, 12))
        self.assertEqual((opponent[5], opponent[24], 15 - sum(opponent)), (4, 2, 9))
        self.assertEqual(to_gnubg_board(state, -1), [own, opponent])
        self.assertEqual(state, snapshot)


@unittest.skipUnless(importlib.util.find_spec("gnubg_nn"), "optional gnubg-nn not installed")
class GNUBGIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.agent = GNUBGAgent(random.Random(1))
        self.g = self.agent.engine

    def test_known_start_position_id(self):
        self.assertEqual(to_gnubg_board(GameState.initial_state()),
                         self.g.board_from_position_id("4HPwATDgc/ABMA"))

    def test_terminal_probabilities_confirm_row_one_perspective(self):
        state = position({1: 1})  # P2 has all 15 off.
        self.assertEqual(self.g.probabilities(to_gnubg_board(state, 1), 0)[0], 0)
        self.assertEqual(self.g.probabilities(to_gnubg_board(state, -1), 0)[0], 1)

    def test_independent_legal_outcomes_for_all_rolls_both_players(self):
        rng = random.Random(119)
        positions = [GameState.initial_state(),
                     position({24: -1, 23: -2, 8: 14}, {1: 1}),
                     position({1: 1, 2: 2, 17: -14}, {-1: 1}),
                     position({6: 3, 3: 5, 1: 2, 19: -4, 22: -3, 24: -5})]
        state = GameState.initial_state()
        for turn in range(25):
            player = state.current_player
            dice = (rng.randint(1, 6), rng.randint(1, 6))
            moves = get_legal_moves(state, dice, player)
            if moves:
                for step in rng.choice(moves).steps:
                    state = apply_step(state, step, player)
            state.current_player *= -1
            if turn % 5 == 0:
                positions.append(state.copy())
        for state in positions:
            for player in (1, -1):
                for d1 in range(1, 7):
                    for d2 in range(d1, 7):
                        with self.subTest(board=state.board, player=player, dice=(d1, d2)):
                            expected = set(self.g.moves(to_gnubg_board(state, player), d1, d2))
                            actual = set()
                            for move in get_legal_moves(state, (d1, d2), player):
                                after = state.copy()
                                for step in move.steps:
                                    after = apply_step(after, step, player)
                                actual.add(self.g.key_of_board(to_gnubg_board(after, player)))
                            self.assertEqual(actual, expected)

    def test_immediate_wins_for_both_players(self):
        for player in (1, -1):
            state = position({3: 1, 22: -1})
            state.current_player = player
            move = self.agent.choose_move(state, (3, 6))
            self.assertIn(move, get_legal_moves(state, (3, 6), player))
            for step in move.steps:
                state = apply_step(state, step, player)
            self.assertEqual(state.borne_off[player], 15)

    def test_pass_reproducibility_and_input_preservation(self):
        blocked = position({24: -2, 23: -2}, {1: 15})
        self.assertIsNone(self.agent.choose_move(blocked, (1, 2)))
        state = GameState.initial_state()
        snapshot = state.copy()
        move = self.agent.choose_move(state, (2, 6))
        self.assertIn(move, get_legal_moves(state, (2, 6), 1))
        self.assertEqual(move, GNUBGAgent(random.Random(1)).choose_move(state, (2, 6)))
        self.assertEqual(state, snapshot)

    def test_win_probability_objective_ignores_gammon_rewards(self):
        state = GameState.initial_state()
        target = get_legal_moves(state, (2, 6), 1)[0]
        after = state.copy()
        for step in target.steps:
            after = apply_step(after, step, 1)
        target_key = self.g.key_of_board(to_gnubg_board(after, -1))

        def probabilities(board, plies):
            self.assertEqual(plies, 0)
            if self.g.key_of_board(board) == target_key:
                return (.1, 0, 0, 0, 0)
            return (.2, 0, 0, .7, 0)  # Opponent gammon losses are ignored.

        with patch.object(self.g, "probabilities", side_effect=probabilities):
            chosen = self.agent.choose_move(state, (2, 6))
        for step in chosen.steps:
            state = apply_step(state, step, 1)
        self.assertEqual(self.g.key_of_board(to_gnubg_board(state, -1)), target_key)

    def test_metadata_records_frozen_external_assets(self):
        meta = self.agent.benchmark_metadata()
        self.assertEqual(meta["version"], "1.1.0a11")
        self.assertEqual(meta["evaluation_plies"], 0)
        self.assertIn("gnubg.weights", meta["data_sha256"])
        self.assertTrue(all(len(value) == 64 for value in meta["data_sha256"].values()))


if __name__ == "__main__":
    unittest.main()
