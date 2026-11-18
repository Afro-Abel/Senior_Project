"""Measure one opening decision; this is not a playing-strength evaluation."""
import argparse
import random
from time import perf_counter

from agents.style_defaults import SELECTED_STYLE_STRENGTHS
from agents.monte_carlo_agent import MonteCarloAgent
from agents.rear_priority_agent import RearPriorityAgent
from agents.front_priority_agent import FrontPriorityAgent
from agents.aggressive_agent import AggressiveAgent
from agents.defensive_agent import DefensiveAgent
from math import isfinite
from backgammon.variants import RULE_VARIANTS, validate_rule_variant
from backgammon.game import BackgammonGame
from monte_carlo.heuristic import load_heuristic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-variant", choices=RULE_VARIANTS, default="standard")
    parser.add_argument("--agent", choices=("baseline", "rear-priority", "front-priority", "aggressive", "defensive"), default="baseline")
    parser.add_argument("--style-strength", type=float, default=None,
                        help="override the selected strength for this agent")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4,
                        help="subsequent player turns per rollout (default: 4)")
    parser.add_argument("--rollout-policy", choices=("random", "heuristic"), default="random")
    parser.add_argument("--heuristic-config", default=None, help="JSON weight/scale overrides")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    if args.style_strength is None:
        args.style_strength = SELECTED_STYLE_STRENGTHS.get(args.agent, 0.0)
    if not isfinite(args.style_strength) or args.style_strength < 0:
        parser.error("--style-strength must be finite and nonnegative")
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.depth < 0:
        parser.error("--depth must be nonnegative")
    try:
        config = load_heuristic_config(args.heuristic_config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    game = BackgammonGame(random.Random(args.seed), args.rule_variant)
    dice = game.roll_opening_dice()
    agent_class = {"baseline": MonteCarloAgent, "rear-priority": RearPriorityAgent,
                   "front-priority": FrontPriorityAgent, "aggressive": AggressiveAgent,
                   "defensive": DefensiveAgent}[args.agent]
    extra = {"style_strength": args.style_strength} if args.agent != "baseline" else {}
    agent = agent_class(args.rollouts, random.Random(args.seed + 1), depth=args.depth,
                        rollout_policy=args.rollout_policy, heuristic_config=config, **extra)
    print(f"Opening: player={game.state.current_player}, dice={dice}; "
          f"{args.rollouts} rollouts per candidate, depth={args.depth} turns, policy={args.rollout_policy}", flush=True)
    start = perf_counter()
    evaluations = agent.evaluate_moves(game.state, dice)
    elapsed = perf_counter() - start
    best = max(evaluations, key=lambda e: (e.immediate_win, e.score))
    total = sum(e.rollouts for e in evaluations)
    print(f"Candidates={len(evaluations)} (distinct resulting positions), "
          f"completed rollouts={total}")
    print(f"Decision evaluation time={elapsed:.3f}s; "
          f"rollouts/second={total / elapsed:.1f}")
    if args.agent != "baseline":
        print(f"Best adjusted score={best.score:.3f}; MC utility={best.mc_score:.3f}; "
              f"style bonus={best.style_adjustment:.3f}; move={best.move}")
    else:
        print(f"Highest mean utility={best.score:.3f}; move={best.move}")

if __name__ == "__main__":
    main()
