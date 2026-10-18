"""Flat Monte Carlo with a configurable turn horizon and shared heuristic."""
from dataclasses import dataclass
import random

from backgammon.move import Move
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState, NUM_CHECKERS
from monte_carlo.rollout import rollout


@dataclass(frozen=True)
class MoveEvaluation:
    move: Move
    score: float
    rollouts: int
    immediate_win: bool = False

class MonteCarloAgent:
    def __init__(self, rollouts_per_move: int = 10,
                 rng: random.Random | None = None, depth: int = 4,
                 rollout_policy: str = "random"):
        if type(rollouts_per_move) is not int or rollouts_per_move < 1:
            raise ValueError("rollouts_per_move must be a positive integer")
        self.rollouts_per_move = rollouts_per_move
        if type(depth) is not int or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        self.depth = depth
        if rollout_policy not in ("random", "heuristic"):
            raise ValueError("rollout_policy must be random or heuristic")
        self.rollout_policy = rollout_policy
        self.rng = rng if rng is not None else random.Random()

    def evaluate_moves(self, state: GameState,
                       dice: tuple[int, int]) -> list[MoveEvaluation]:
        """Average bounded rollout utilities from the current player's perspective.

        Each nonterminal candidate gets the same simulation budget. Immediate
        wins are exact outcomes and need no simulations. Depth counts subsequent
        player turns, excluding the candidate. Scores are not win probabilities.
        """
        if any(count == NUM_CHECKERS for count in state.borne_off.values()):
            raise ValueError("Cannot evaluate moves in a finished game")
        player = state.current_player
        evaluations = []
        seen_positions = set()
        for move in get_legal_moves(state, dice, player):
            after = state.copy()
            for step in move.steps:
                after = apply_step(after, step, player)
            # Preserve the first legal sequence for each complete resulting state.
            # Include bar/off counts as well as the board: hits and bearing off
            # are part of the outcome. Candidate order stays deterministic.
            key = (tuple(after.board),
                   tuple(after.bar[p] for p in (1, -1)),
                   tuple(after.borne_off[p] for p in (1, -1)),
                   after.current_player)
            if key in seen_positions:
                continue
            seen_positions.add(key)
            if after.borne_off[player] == NUM_CHECKERS:
                evaluations.append(MoveEvaluation(move, 1.0, 0, immediate_win=True))
                continue
            # The candidate is a whole turn, so the opponent rolls next.
            after.current_player = -player
            # At depth zero the cutoff score is deterministic, so evaluate once.
            count = self.rollouts_per_move if self.depth else 1
            score = sum(rollout(after, self.rng, self.depth, player, self.rollout_policy)
                        for _ in range(count)) / count
            evaluations.append(MoveEvaluation(move, score, count))
        return evaluations

    def choose_move(self, state: GameState, dice: tuple[int, int]) -> Move | None:
        """Return a legal move, or None for a forced pass. Randomize tied scores."""
        evaluations = self.evaluate_moves(state, dice)
        if not evaluations:
            return None
        best = max((e.score, e.immediate_win) for e in evaluations)
        return self.rng.choice([e.move for e in evaluations
                                if (e.score, e.immediate_win) == best])
