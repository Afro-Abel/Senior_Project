"""Depth-limited rollouts with random or greedy heuristic move selection."""
import random

from backgammon.game import BackgammonGame
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC, HeuristicConfig, evaluate_position


def choose_heuristic_move(state, moves, rng, config=DEFAULT_HEURISTIC):
    """Greedy one-turn evaluation for the acting player; randomize tied outcomes.

    No nested search. Equivalent complete states receive one candidate each.
    """
    player = state.current_player
    seen = set()
    candidates = []
    for move in moves:
        after = state.copy()
        for step in move.steps:
            after = apply_step(after, step, player)
        key = (tuple(after.board), tuple(after.bar[p] for p in (1, -1)),
               tuple(after.borne_off[p] for p in (1, -1)))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((evaluate_position(after, player, config), move))
    if not candidates:
        return None
    best = max(score for score, _ in candidates)
    return rng.choice([move for score, move in candidates if score == best])


def rollout(state: GameState, rng: random.Random,
            depth: int, player: int, policy: str = "random",
            config: HeuristicConfig = DEFAULT_HEURISTIC) -> float:
    """Simulate at most depth player turns and score from player's perspective.

    This continues an existing game: there is no new opening roll. A blocked
    player passes. Random choices are uniform over legal move sequences;
    heuristic choices maximize the acting player's immediate position score.
    A forced pass counts as a turn. Depth zero evaluates the input immediately.
    """
    if type(depth) is not int or depth < 0:
        raise ValueError("depth must be a nonnegative integer")
    if player not in (1, -1):
        raise ValueError("player must be 1 or -1")
    if policy not in ("random", "heuristic"):
        raise ValueError("policy must be random or heuristic")
    game = BackgammonGame(rng=rng)
    game.state = state.copy()
    for _ in range(depth):
        if game.is_game_over():
            break
        moves = get_legal_moves(game.state, game.roll_dice(),
                                game.state.current_player)
        if moves:
            move = (rng.choice(moves) if policy == "random" else
                    choose_heuristic_move(game.state, moves, rng, config))
            game.apply_move(move)
        if not game.is_game_over():
            game.switch_player()
    return evaluate_position(game.state, player, config)
