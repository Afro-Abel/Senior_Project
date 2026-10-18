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
    def __init__(self, rng: random.Random | None = None):
        self.state = GameState.initial_state()
        self.rng = rng if rng is not None else random

    def roll_dice(self) -> tuple[int, int]:
        """
        Roll two six-sided dice.
        """
        return (
            self.rng.randint(1, 6),
            self.rng.randint(1, 6),
        )

    def roll_opening_dice(self) -> tuple[int, int]:
        """Call once at the start of a game, before playing any turns.

        Each player rolls one die (P1 first, P2 second). Reroll ties,
        set the higher roller as current player, and return both dice
        for that player's first move. Do not roll again on the first turn.
        """
        while True:
            dice = self.roll_dice()
            if dice[0] != dice[1]:
                self.state.current_player = (
                    PLAYER_1 if dice[0] > dice[1] else PLAYER_2
                )
                return dice

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
