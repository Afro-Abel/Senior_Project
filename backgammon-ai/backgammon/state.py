from __future__ import annotations

from dataclasses import dataclass, field


PLAYER_1 = 1
PLAYER_2 = -1
NUM_POINTS = 24
NUM_CHECKERS = 15


@dataclass
class GameState:

    board: list[int] = field(default_factory=lambda: [0] * NUM_POINTS)

    bar: dict[int, int] = field(
        default_factory=lambda: {
            PLAYER_1: 0,
            PLAYER_2: 0,
        }
    )

    borne_off: dict[int, int] = field(
        default_factory=lambda: {
            PLAYER_1: 0,
            PLAYER_2: 0,
        }
    )
    current_player: int = PLAYER_1

    @classmethod
    def initial_state(cls) -> "GameState":
        """
        Create the standard Backgammon starting position.

        Player 1 moves from point 24 toward point 1.
        Player 2 moves from point 1 toward point 24.

        board[0]  = point 1
        board[23] = point 24

        Positive values represent Player 1 checkers.
        Negative values represent Player 2 checkers.
        """
        state = cls()

        # Player 1
        state.board[23] = 2   # Point 24
        state.board[12] = 5   # Point 13
        state.board[7] = 3    # Point 8
        state.board[5] = 5    # Point 6

        # Player 2
        state.board[0] = -2   # Point 1
        state.board[11] = -5  # Point 12
        state.board[16] = -3  # Point 17
        state.board[18] = -5  # Point 19

        return state

    def copy(self) -> "GameState":
        """
        Return an independent copy of this game state.
        """
        return GameState(
            board=self.board.copy(),
            bar=self.bar.copy(),
            borne_off=self.borne_off.copy(),
            current_player=self.current_player,
        )

    def checker_count(self, player: int) -> int:
        """
        Return the total number of checkers belonging to a player,
        including checkers on the board, on the bar, and borne off.
        """
        board_count = sum(abs(value) for value in self.board if value * player > 0)

        return (
            board_count
            + self.bar[player]
            + self.borne_off[player])