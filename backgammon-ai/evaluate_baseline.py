"""Reproducible baseline games against random, heuristic-only, or gnubg-nn play with correctness and timing checks."""
import argparse
from dataclasses import asdict
import hashlib
import json
from math import ceil, sqrt, isfinite
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

from agents.style_defaults import SELECTED_STYLE_STRENGTHS
from agents.monte_carlo_agent import MonteCarloAgent
from agents.random_agent import RandomAgent
from agents.rear_priority_agent import RearPriorityAgent
from agents.front_priority_agent import FrontPriorityAgent
from agents.aggressive_agent import AggressiveAgent
from agents.defensive_agent import DefensiveAgent
from agents.gnubg_agent import GNUBGAgent
from backgammon.variants import RULE_VARIANTS, validate_rule_variant
from backgammon.game import BackgammonGame
from backgammon.rules import get_legal_moves
from monte_carlo.heuristic import DEFAULT_HEURISTIC, load_heuristic_config
from behavior_metrics import observe_turn, summarize_turns
from validate_random_games import check_state, dice_usage_checker


def wilson_interval(wins, games):
    """Approximate 95% Wilson interval for independent binary game outcomes."""
    z = 1.959963984540054
    p = wins / games
    denominator = 1 + z * z / games
    center = (p + z * z / (2 * games)) / denominator
    margin = z * sqrt(p * (1 - p) / games + z * z / (4 * games ** 2)) / denominator
    return [max(0, center - margin), min(1, center + margin)]


def play_game(index, seed, rollouts, depth, max_turns, opponent_kind="random", rollout_policy="random", heuristic_config=DEFAULT_HEURISTIC, agent_kind="baseline", style_strength=None, rule_variant="standard", collect_rule_impact=False):
    validate_rule_variant(rule_variant)
    if rule_variant != "standard" and opponent_kind == "gnubg":
        raise ValueError("gnubg evaluation assumes standard rules")
    if opponent_kind not in ("random", "heuristic", "gnubg", "baseline", *SELECTED_STYLE_STRENGTHS):
        raise ValueError("opponent_kind must be random, heuristic, gnubg, or baseline")
    if agent_kind not in ("baseline", "rear-priority", "front-priority", "aggressive", "defensive"):
        raise ValueError("agent_kind must be baseline, rear-priority, front-priority, aggressive, or defensive")
    # Independent streams per game prevent search settings changing actual dice.
    game_seed = seed + index * 3
    game = BackgammonGame(random.Random(game_seed), rule_variant)
    candidate_class = {"baseline": MonteCarloAgent, "rear-priority": RearPriorityAgent,
                       "front-priority": FrontPriorityAgent, "aggressive": AggressiveAgent,
                       "defensive": DefensiveAgent}[agent_kind]
    extra = {"style_strength": style_strength} if agent_kind != "baseline" else {}
    mc = candidate_class(rollouts, random.Random(game_seed + 1), depth=depth,
                         rollout_policy=rollout_policy, heuristic_config=heuristic_config, **extra)
    opponent_rng = random.Random(game_seed + 2)
    opponent = (RandomAgent(opponent_rng) if opponent_kind == "random" else
                GNUBGAgent(opponent_rng) if opponent_kind == "gnubg" else
                MonteCarloAgent(20, opponent_rng, depth=4, rollout_policy="heuristic")
                if opponent_kind == "baseline" else
                MonteCarloAgent(1, opponent_rng, depth=0))
    if opponent_kind in SELECTED_STYLE_STRENGTHS:
        opponent_class = {"rear-priority": RearPriorityAgent, "front-priority": FrontPriorityAgent,
                          "aggressive": AggressiveAgent, "defensive": DefensiveAgent}[opponent_kind]
        opponent = opponent_class(rollouts, opponent_rng, depth=depth,
                                  rollout_policy=rollout_policy, heuristic_config=heuristic_config)
    opponent_times, opponent_style_metrics = [], []
    impact_counts = {"turns_checked": 0, "turns_with_changed_outcomes": 0}
    mc_player = 1 if index % 2 == 0 else -1
    opening = game.roll_opening_dice()
    starter = game.state.current_player
    decisions = []
    style_metrics = []
    behavior_turns = []
    passes = hits = 0
    check_state(game.state)
    for turn in range(1, max_turns + 1):
        player = game.state.current_player
        dice = opening if turn == 1 else game.roll_dice()
        snapshot = game.state.copy()
        try:
            legal = get_legal_moves(game.state, dice, player)
            if collect_rule_impact:
                from rule_impact import compare_rule_outcomes
                impact_counts["turns_checked"] += 1
                impact_counts["turns_with_changed_outcomes"] += compare_rule_outcomes(snapshot, dice, player, legal)

            if player == mc_player:
                start = perf_counter()
                move = mc.choose_move(game.state, dice)
                elapsed = perf_counter() - start
                # Include forced-pass decisions; report them explicitly in metadata.
                decisions.append(elapsed)
                if agent_kind != "baseline" and mc.last_decision_metrics is not None:
                    style_metrics.append(mc.last_decision_metrics.copy())
            else:
                if opponent_kind != "random":
                    start = perf_counter()
                    move = opponent.choose_move(game.state, dice)
                    if opponent_kind in SELECTED_STYLE_STRENGTHS:
                        opponent_times.append(perf_counter() - start)
                        if opponent.last_decision_metrics is not None:
                            opponent_style_metrics.append(opponent.last_decision_metrics.copy())
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
            observed = observe_turn(snapshot, game.state, move, legal, player)
            observed.update({"turn": turn, "player": player,
                             "role": "candidate" if player == mc_player else "opponent"})
            behavior_turns.append(observed)
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
                        "decision_seconds": decisions, "agent": agent_kind, "rule_variant": rule_variant,
                        "style_decisions": style_metrics, "behavior_turns": behavior_turns,
                        **({"opponent_decision_seconds": opponent_times,
                            "opponent_style_decisions": opponent_style_metrics} if opponent_kind in SELECTED_STYLE_STRENGTHS else {}),
                        **({"rule_impact": impact_counts} if collect_rule_impact else {})}
            game.switch_player()
        except Exception as error:
            raise RuntimeError(f"game={index + 1}, seed={game_seed}, turn={turn}, "
                               f"player={player}, dice={dice}, state={asdict(snapshot)}") from error
    raise RuntimeError(f"Game {index + 1}, seed={game_seed} exceeded {max_turns} turns")


def summarize(records, opponent_kind="random"):
    wins = sum(r["mc_won"] for r in records)
    times = sorted(t for r in records for t in r["decision_seconds"])
    result = {"games": len(records), "mc_wins": wins,
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
    style = [d for r in records for d in r.get("style_decisions", [])]
    if style:
        opportunities = [d for d in style if d["preference_range"] > 0]
        result["style_behavior"] = {
            "style": style[0]["style"], "evaluated_decisions": len(style),
            "decisions_with_different_style_options": len(opportunities),
            "mean_normalized_preference_when_options_differ": (
                mean(d["chosen_normalized_preference"] for d in opportunities)
                if opportunities else None),
            "style_overrode_mc_decisions": sum(d["style_overrode_mc"] for d in style),
            "mean_mc_utility_sacrificed": mean(d["mc_utility_sacrificed"] for d in style)}
    observations = [t for r in records for t in r.get("behavior_turns", [])]
    if observations:
        result["gameplay_behavior"] = {
            role: summarize_turns([t for t in observations if t["role"] == role])
            for role in ("candidate", "opponent")}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-variant", choices=RULE_VARIANTS, default="standard")
    parser.add_argument("--agent", choices=("baseline", "rear-priority", "front-priority", "aggressive", "defensive"), default="baseline")
    parser.add_argument("--style-strength", type=float, default=None,
                        help="override the selected strength for this agent")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--opponent", choices=("random", "heuristic", "gnubg", "baseline"), default="random",
                        help="random, default depth-zero heuristic, external gnubg, or frozen original MC baseline")
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--rollout-policy", choices=("random", "heuristic"), default="random")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--heuristic-config", type=Path, default=None,
                        help="JSON overrides for the candidate only; opponents remain frozen")
    parser.add_argument("--max-turns", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.style_strength is None:
        args.style_strength = SELECTED_STYLE_STRENGTHS.get(args.agent, 0.0)
    if not isfinite(args.style_strength) or args.style_strength < 0:
        parser.error("--style-strength must be finite and nonnegative")
    if args.rule_variant != "standard" and args.opponent == "gnubg":
        parser.error("gnubg evaluation assumes standard rules")
    if args.games < 2 or args.games % 2:
        parser.error("--games must be positive and even (at least 2) for balanced seats")
    if args.rollouts < 1 or args.depth < 0 or args.max_turns < 1:
        parser.error("rollouts/max-turns must be positive; depth must be nonnegative")
    if args.output is None:
        args.output = (Path(f"data/{args.agent}_vs_{args.opponent}.json") if args.agent != "baseline"
                       else Path("data/baseline_validation.json" if args.opponent == "random"
                                 else f"data/baseline_vs_{args.opponent}.json"))
        if args.rule_variant != "standard":
            args.output = args.output.with_name(f"{args.output.stem}_{args.rule_variant}.json")
    try:
        heuristic_config = load_heuristic_config(args.heuristic_config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    started = perf_counter()
    records = []
    for index in range(args.games):
        record = play_game(index, args.seed, args.rollouts, args.depth, args.max_turns, args.opponent, args.rollout_policy, heuristic_config, args.agent, args.style_strength, args.rule_variant)
        records.append(record)
        print(f"Game {index + 1}/{args.games}: MC={'win' if record['mc_won'] else 'loss'}, "
              f"seat={record['mc_player']}, turns={record['turns']}", flush=True)
    root = Path(__file__).resolve().parent
    sources = ["agents/monte_carlo_agent.py", "agents/random_agent.py", "agents/gnubg_agent.py",
               "agents/styled_agent.py", "agents/rear_priority_agent.py", "agents/front_priority_agent.py", "agents/aggressive_agent.py", "agents/defensive_agent.py",
               "monte_carlo/heuristic.py", "monte_carlo/rollout.py",
               "backgammon/variants.py", "agents/style_defaults.py", "backgammon/rules.py", "backgammon/game.py", "backgammon/state.py",
               "backgammon/move.py", "evaluate_baseline.py", "validate_random_games.py", "behavior_metrics.py"]
    report = {"settings": {**vars(args), "output": str(args.output),
                           "heuristic_config": asdict(heuristic_config),
                           "heuristic_config_path": (str(args.heuristic_config)
                                                     if args.heuristic_config else None)},
              "style_settings": ({"name": args.agent, "strength": args.style_strength,
                                   "raw_preference": ("decrease in own sum of squared distances to off"
                                                      if args.agent == "rear-priority" else
                                                      "52 * pip progress minus squared-distance reduction"
                                                      if args.agent == "front-priority" else
                                                      "opponent bar increase during the candidate action"
                                                      if args.agent == "aggressive" else
                                                      "own direct-threat reduction plus own made-point increase while contact remains"),
                                   "normalization": "min-max over distinct candidate outcomes; zero if equal",
                                   "scope": "candidate score only; cutoff and rollout policy shared"}
                                  if args.agent != "baseline" else None),
              "opponent_settings": ({"heuristic_config": asdict(DEFAULT_HEURISTIC),
                                     "depth": 4 if args.opponent == "baseline" else 0,
                                     "rollouts": 20 if args.opponent == "baseline" else 1,
                                     "rollout_policy": "heuristic"}
                                    if args.opponent in ("heuristic", "baseline") else None),
              "source_sha256": {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                for p in sources},
              "notes": ["Independent game seeds; alternating MC seats; standard opening contest.",
                        ("Random opponent samples legal sequences, not unique resulting states."
                         if args.opponent == "random" else
                         "External gnubg-nn ranks unique outcomes by single-game win probability at 0 ply."
                         if args.opponent == "gnubg" else
                         "Frozen original MC: default weights, 20 guided rollouts, depth four."
                         if args.opponent == "baseline" else
                         "Heuristic opponent uses the same cutoff evaluator at depth zero, "
                         "with unique resulting positions and random tie breaking."),
                        "Timing covers MC choose_move, including forced passes, excluding validation.",
                        "Wilson interval treats game outcomes as independent; pilot evidence only.",
                        "Rollout policy applies to both simulated players; heuristic weights remain fixed.",
                        "This evaluates current fixed heuristic weights, not their optimality."],
              "external_benchmark": (GNUBGAgent().benchmark_metadata()
                                     if args.opponent == "gnubg" else None),
              "summary": summarize(records, args.opponent), "games": records,
              "elapsed_seconds": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
