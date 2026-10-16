import random

from backgammon.state import (
    GameState,
    PLAYER_1,
    PLAYER_2,
    NUM_CHECKERS,
)
from backgammon.move import Move
from backgammon.rules import apply_step


class BackgammonGame:
    def __init__(self):
        self.state = GameState.initial_state()

    def roll_dice(self) -> tuple[int, int]:
        """
        Roll two six-sided dice.
        """
        return (
            random.randint(1, 6),
            random.randint(1, 6),
        )

    def apply_move(self, move: Move) -> None:
        """
        Apply a complete move to the current game state.
        """
        for step in move.steps:
            self.state = apply_step(
                self.state,
                step,
                self.state.current_player,
            )

    def switch_player(self) -> None:
        """
        Switch the current player.
        """
        self.state.current_player *= -1

    def is_game_over(self) -> bool:
        """
        Return True if either player has borne off all checkers.
        """
        return (
            self.state.borne_off[PLAYER_1] == NUM_CHECKERS
            or
            self.state.borne_off[PLAYER_2] == NUM_CHECKERS
        )

    def get_winner(self) -> int | None:
        """
        Return the winning player, or None if the game is not over.
        """
        if self.state.borne_off[PLAYER_1] == NUM_CHECKERS:
            return PLAYER_1

        if self.state.borne_off[PLAYER_2] == NUM_CHECKERS:
            return PLAYER_2

        return None