"""Measure one opening decision; this is not a playing-strength evaluation."""
import argparse
import random
from time import perf_counter

from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.game import BackgammonGame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4,
                        help="subsequent player turns per rollout (default: 4)")
    parser.add_argument("--rollout-policy", choices=("random", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.depth < 0:
        parser.error("--depth must be nonnegative")
    game = BackgammonGame(random.Random(args.seed))
    dice = game.roll_opening_dice()
    agent = MonteCarloAgent(args.rollouts, random.Random(args.seed + 1), depth=args.depth, rollout_policy=args.rollout_policy)
    print(f"Opening: player={game.state.current_player}, dice={dice}; "
          f"{args.rollouts} rollouts per candidate, depth={args.depth} turns, policy={args.rollout_policy}", flush=True)
    start = perf_counter()
    evaluations = agent.evaluate_moves(game.state, dice)
    elapsed = perf_counter() - start
    best = max(evaluations, key=lambda e: e.score)
    total = sum(e.rollouts for e in evaluations)
    print(f"Candidates={len(evaluations)} (distinct resulting positions), "
          f"completed rollouts={total}")
    print(f"Decision evaluation time={elapsed:.3f}s; "
          f"rollouts/second={total / elapsed:.1f}")
    print(f"Highest mean utility={best.score:.3f}; "
          f"move={best.move}")

if __name__ == "__main__":
    main()
