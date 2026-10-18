"""Seeded engine smoke test. Run: python3 validate_random_games.py --games 100.

Random play samples move sequences, not distinct resulting board positions.
This validates engine execution; its results are not agent-strength estimates.
"""
import argparse
from functools import lru_cache
import random

from agents.random_agent import RandomAgent
from backgammon.game import BackgammonGame
from backgammon.rules import apply_step, get_legal_moves, get_legal_steps
from backgammon.state import GameState


def check_state(state):
    if len(state.board) != 24 or state.current_player not in (1, -1):
        raise AssertionError("Invalid board size or current player")
    for player in (1, -1):
        if not 0 <= state.bar[player] <= 15:
            raise AssertionError("Invalid bar count")
        if not 0 <= state.borne_off[player] <= 15:
            raise AssertionError("Invalid borne-off count")
        if state.checker_count(player) != 15:
            raise AssertionError("Checker count changed")


def dice_usage_checker(state, dice, player):
    """Explore remaining dice in any order, independently of full-move generation.

    Scores prefer the most dice used, then the higher die. Single-step rules
    are shared with the engine and are covered separately by rule fixtures.
    """
    @lru_cache(maxsize=None)
    def best(board, bar, off, remaining):
        current = GameState(list(board), dict(zip((1, -1), bar)),
                            dict(zip((1, -1), off)), player)
        score = (0, 0)
        for die in set(remaining):
            rest = list(remaining)
            rest.remove(die)
            for step in get_legal_steps(current, die, player):
                after = apply_step(current, step, player)
                tail = best(*key(after), tuple(rest))
                score = max(score, (1 + tail[0], die + tail[1]))
        return score

    def key(current):
        return (tuple(current.board), tuple(current.bar[p] for p in (1, -1)),
                tuple(current.borne_off[p] for p in (1, -1)))

    expanded = tuple(sorted(dice if dice[0] != dice[1] else dice * 2))
    optimum = best(*key(state), expanded)

    def check_move(move):
        def match(current, steps, remaining, used):
            if not steps:
                return (len(move.steps), used) == optimum
            for die in set(remaining):
                if steps[0] in get_legal_steps(current, die, player):
                    rest = list(remaining)
                    rest.remove(die)
                    after = apply_step(current, steps[0], player)
                    check_state(after)
                    if match(after, steps[1:], rest, used + die):
                        return True
            return False
        if not match(state, move.steps, list(expanded), 0):
            raise AssertionError(f"Illegal steps or forced-dice usage: {move}")

    return optimum, check_move


def run_games(games=100, seed=20261016, max_turns=10000):
    dice_rng = random.Random(seed)
    agent = RandomAgent(random.Random(seed + 1))
    wins = {1: 0, -1: 0}
    lengths = []
    passes = hits = entries = 0
    for game_index in range(games):
        game = BackgammonGame(rng=dice_rng)
        opening_dice = game.roll_opening_dice()
        check_state(game.state)
        for turn in range(1, max_turns + 1):
            try:
                player = game.state.current_player
                dice = opening_dice if turn == 1 else game.roll_dice()
                snapshot = game.state.copy()
                moves = get_legal_moves(game.state, dice, player)
                if game.state != snapshot:
                    raise AssertionError("Move generation mutated its input")
                optimum, check_move = dice_usage_checker(game.state, dice, player)
                if moves:
                    move = agent.choose_move(moves)
                    check_move(move)
                    before_bar = game.state.bar.copy()
                    game.apply_move(move)
                    hits += game.state.bar[-player] - before_bar[-player]
                    entries += before_bar[player] - game.state.bar[player]
                else:
                    if optimum[0] != 0:
                        raise AssertionError("Passed despite a playable die")
                    passes += 1
                check_state(game.state)
                if game.is_game_over():
                    winner = game.get_winner()
                    if winner not in (1, -1) or game.state.borne_off[winner] != 15:
                        raise AssertionError("Invalid winner")
                    wins[winner] += 1
                    lengths.append(turn)
                    break
                game.switch_player()
            except Exception as error:
                raise AssertionError(
                    f"seed={seed}, game={game_index + 1}, turn={turn}, "
                    f"player={player}, dice={dice}: {error}"
                ) from error
        else:
            raise AssertionError(f"Game {game_index + 1} exceeded {max_turns} turns")
        if (game_index + 1) % 10 == 0:
            print(f"Validated {game_index + 1}/{games} games", flush=True)
    print(f"Passed: {games} games; P1 wins={wins[1]}, P2 wins={wins[-1]}")
    print(f"Turns: min={min(lengths)}, mean={sum(lengths) / games:.1f}, "
          f"max={max(lengths)}")
    print(f"Hits={hits}, bar entries={entries}, forced passes={passes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--max-turns", type=int, default=10000)
    args = parser.parse_args()
    if args.games < 1 or args.max_turns < 1:
        parser.error("--games and --max-turns must be positive")
    run_games(args.games, args.seed, args.max_turns)
