import random
import unittest
from unittest.mock import Mock, patch

from backgammon.game import BackgammonGame
from backgammon.rules import get_legal_moves
import main as terminal


class OpeningTests(unittest.TestCase):
    def test_higher_roller_starts_and_board_is_preserved(self):
        for dice, starter in [((6, 2), 1), ((1, 5), -1)]:
            with self.subTest(dice=dice):
                rng = Mock()
                rng.randint.side_effect = dice
                game = BackgammonGame(rng=rng)
                before = game.state.copy()
                self.assertEqual(game.roll_opening_dice(), dice)
                before.current_player = starter
                self.assertEqual(game.state, before)
                self.assertEqual(rng.randint.call_count, 2)
                rng.randint.assert_called_with(1, 6)

    def test_repeated_ties_are_rerolled_and_regular_roll_follows(self):
        rng = Mock()
        rng.randint.side_effect = [3, 3, 6, 6, 2, 5, 4, 4]
        game = BackgammonGame(rng=rng)
        self.assertEqual(game.roll_opening_dice(), (2, 5))
        self.assertEqual(game.state.current_player, -1)
        self.assertEqual(rng.randint.call_count, 6)
        game.switch_player()
        self.assertEqual(game.state.current_player, 1)
        self.assertEqual(game.roll_dice(), (4, 4))
        self.assertEqual(game.state.current_player, 1)

    def test_seeded_openings_are_reproducible(self):
        first = BackgammonGame(random.Random(42))
        second = BackgammonGame(random.Random(42))
        self.assertEqual(first.roll_opening_dice(), second.roll_opening_dice())
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.roll_dice(), second.roll_dice())

    def test_terminal_uses_opening_dice_then_rolls_on_next_turn(self):
        rng = Mock()
        rng.randint.side_effect = [2, 5, 3, 4]
        game = BackgammonGame(rng=rng)
        # Play two real turns, then stop without running a whole interactive game.
        with patch.object(game, "is_game_over", side_effect=[False, False, True]), \
             patch.object(game, "get_winner", return_value=1), \
             patch.object(terminal, "BackgammonGame", return_value=game), \
             patch.object(terminal, "get_legal_moves", wraps=get_legal_moves) as moves, \
             patch("builtins.input", return_value="0"), \
             patch("builtins.print"):
            terminal.main()
        self.assertEqual(moves.call_count, 2)
        self.assertEqual(moves.call_args_list[0].args[1:], ((2, 5), -1))
        self.assertEqual(moves.call_args_list[1].args[1:], ((3, 4), 1))
        self.assertEqual(rng.randint.call_count, 4)


if __name__ == "__main__":
    unittest.main()
