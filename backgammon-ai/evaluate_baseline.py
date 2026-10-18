"""Reproducible baseline games against random or heuristic-only play with correctness and timing checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from math import ceil, sqrt
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from agents.monte_carlo_agent import MonteCarloAgent
from agents.random_agent import RandomAgent
from backgammon.game import BackgammonGame
from backgammon.rules import get_legal_moves
from validate_random_games import check_state, dice_usage_checker


def wilson_interval(wins, games):
    """Approximate 95% Wilson interval for independent binary game outcomes."""
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = z * sqrt(p * (1 - p) / games + z * z / (4 * games ** 2)) / denominator
    return [max(0, center - margin), min(1, center + margin)]


def play_game(index, seed, rollouts, depth, max_turns, opponent_kind="random", rollout_policy="random"):
    if opponent_kind not in ("random", "heuristic"):
        raise ValueError("opponent_kind must be random or heuristic")
    # Independent streams per game prevent search settings changing actual dice.
    game_seed = seed + index * 3
    game = BackgammonGame(random.Random(game_seed))
    mc = MonteCarloAgent(rollouts, random.Random(game_seed + 1), depth=depth, rollout_policy=rollout_policy)
    opponent_rng = random.Random(game_seed + 2)
    opponent = (RandomAgent(opponent_rng) if opponent_kind == "random" else
                MonteCarloAgent(1, opponent_rng, depth=0))
    mc_player = 1 if index % 2 == 0 else -1
    opening = game.roll_opening_dice()
    starter = game.state.current_player
    decisions = []
    passes = hits = 0
    check_state(game.state)
    for turn in range(1, max_turns + 1):
        player = game.state.current_player
        dice = opening if turn == 1 else game.roll_dice()
        snapshot = game.state.copy()
        try:
            legal = get_legal_moves(game.state, dice, player)
            if player == mc_player:
                start = perf_counter()
                move = mc.choose_move(game.state, dice)
                elapsed = perf_counter() - start
                # Include forced-pass decisions; report them explicitly in metadata.
                decisions.append(elapsed)
            else:
                if opponent_kind == "heuristic":
                    move = opponent.choose_move(game.state, dice)
                else:
                    move = opponent.choose_move(legal) if legal else None
            if game.state != snapshot:
                raise AssertionError("Agent or move generation mutated the input")
            optimum, check_move = dice_usage_checker(game.state, dice, player)
            if legal:
                if move not in legal:
                    raise AssertionError("Agent chose an illegal move or passed")
                check_move(move)  # Includes conservation after individual steps.
                before_bar = game.state.bar[-player]
                game.apply_move(move)
                hits += game.state.bar[-player] - before_bar
            else:
                if move is not None or optimum[0]:
                    raise AssertionError("Invalid forced pass")
                passes += 1
            check_state(game.state)
            if game.is_game_over():
                winner = game.get_winner()
                if winner not in (1, -1) or game.state.borne_off[winner] != 15:
                    raise AssertionError("Invalid terminal result")
                return {"game": index + 1, "dice_seed": game_seed,
                        "mc_seed": game_seed + 1, "opponent_seed": game_seed + 2,
                        "opponent": opponent_kind, "rollout_policy": rollout_policy,
                        "mc_player": mc_player, "starter": starter,
                        "winner": winner, "mc_won": winner == mc_player,
                        "turns": turn, "hits": hits, "forced_passes": passes,
                        "decision_seconds": decisions}
            game.switch_player()
        except Exception as error:
            raise RuntimeError(f"game={index + 1}, seed={game_seed}, turn={turn}, "
                               f"player={player}, dice={dice}, state={asdict(snapshot)}") from error
    raise RuntimeError(f"Game {index + 1}, seed={game_seed} exceeded {max_turns} turns")


def summarize(records, opponent_kind="random"):
    wins = sum(r["mc_won"] for r in records)
    times = sorted(t for r in records for t in r["decision_seconds"])
    return {"games": len(records), "mc_wins": wins,
            "opponent": opponent_kind, "opponent_wins": len(records) - wins,
            "mc_win_rate": wins / len(records),
            "mc_win_rate_95pct_wilson": wilson_interval(wins, len(records)),
            "by_mc_player": {str(p): {
                "games": sum(r["mc_player"] == p for r in records),
                "wins": sum(r["mc_won"] and r["mc_player"] == p for r in records)
            } for p in (1, -1)},
            "mc_started": sum(r["starter"] == r["mc_player"] for r in records),
            "mean_turns": mean(r["turns"] for r in records),
            "mc_decisions": len(times), "mean_decision_seconds": mean(times),
            "p95_decision_seconds": times[ceil(.95 * len(times)) - 1],
            "max_decision_seconds": max(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--opponent", choices=("random", "heuristic"), default="random",
                        help="heuristic uses the same agent at depth zero")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rollout-policy", choices=("random", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be positive and even (at least 2) for balanced seats")
    if args.rollouts < 1 or args.depth < 0 or args.max_turns < 1:
        parser.error("rollouts/max-turns must be positive; depth must be nonnegative")
    if args.output is None:
        args.output = Path("data/baseline_validation.json" if args.opponent == "random"
                           else "data/baseline_vs_heuristic.json")
    started = perf_counter()
    records = []
    for index in range(args.games):
        record = play_game(index, args.seed, args.rollouts, args.depth, args.max_turns, args.opponent, args.rollout_policy)
        records.append(record)
        print(f"Game {index + 1}/{args.games}: MC={'win' if record['mc_won'] else 'loss'}, "
              f"seat={record['mc_player']}, turns={record['turns']}", flush=True)
    root = Path(__file__).resolve().parent
    sources = ["agents/monte_carlo_agent.py", "agents/random_agent.py",
               "monte_carlo/heuristic.py", "monte_carlo/rollout.py",
               "backgammon/rules.py", "backgammon/game.py", "backgammon/state.py",
               "backgammon/move.py", "evaluate_baseline.py", "validate_random_games.py"]
    report = {"settings": {**vars(args), "output": str(args.output)},
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                for p in sources},
              "notes": ["Independent game seeds; alternating MC seats; standard opening contest.",
                        ("Random opponent samples legal sequences, not unique resulting states."
                         if args.opponent == "random" else
                         "Heuristic opponent uses the same cutoff evaluator at depth zero, "
                         "with unique resulting positions and random tie breaking."),
                        "Timing covers MC choose_move, including forced passes, excluding validation.",
                        "Wilson interval treats game outcomes as independent; pilot evidence only.",
                        "Rollout policy applies to both simulated players; heuristic weights remain fixed.",
                        "This evaluates current fixed heuristic weights, not their optimality."],
              "summary": summarize(records, args.opponent), "games": records,
              "elapsed_seconds": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
