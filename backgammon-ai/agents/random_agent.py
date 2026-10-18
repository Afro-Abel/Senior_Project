"""A reproducible baseline that samples uniformly from legal move sequences."""
import random

from backgammon.move import Move


class RandomAgent:
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng if rng is not None else random.Random()

    def choose_move(self, legal_moves: list[Move]) -> Move:
        if not legal_moves:
            raise ValueError("Cannot choose a move when no legal moves exist")
        return self.rng.choice(legal_moves)
