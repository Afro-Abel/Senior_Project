"""Held-out confirmation: borne-off weights 0 and 4 against the external reference."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from math import comb

from agents.gnubg_agent import GNUBGAgent
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from evaluate_baseline import play_game, summarize
from monte_carlo.heuristic import DEFAULT_HEURISTIC, HeuristicConfig


WEIGHTS = (0, 4)
SEEDS = (1000000, 1000100, 1000200, 1000300, 1000400)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_game(job):
    weight, seed, index, settings = job
    start = perf_counter()
    config = replace(DEFAULT_HEURISTIC, borne_off=weight)
    result = play_game(index, seed, settings["rollouts"], settings["depth"],
                       settings["max_turns"], "gnubg", "heuristic", config)
    return {"weight": weight, "batch_seed": seed, "index": index,
            "heuristic_config": asdict(config), "game": result,
            "elapsed_seconds": perf_counter() - start}


def results_summary(results, settings):
    groups = []
    for weight in settings["weights"]:
        selected = [r for r in results if r["weight"] == weight]
        if selected:
            groups.append({"borne_off": weight, "completed_games": len(selected),
                           "summary": summarize([r["game"] for r in selected], "gnubg"),
                           "game_elapsed_seconds": sum(r["elapsed_seconds"] for r in selected)})
    expected = len(settings["weights"]) * len(settings["seeds"]) * settings["games_per_batch"]
    return {"status": "complete" if len(results) == expected else "running",
            "completed_games": len(results), "expected_games": expected,
            "note": "Held-out comparison of selected weight 0 versus original 4. Do not select from partial results; defaults are not changed automatically.",
            "candidates": groups,
            "matched_comparison": paired_comparison(results) if len(results) == expected else None}


def paired_comparison(results):
    """Compare matched RNG initializations; game trajectories can differ."""
    groups = {weight: {(r["batch_seed"], r["index"]): r["game"]["mc_won"]
                       for r in results if r["weight"] == weight} for weight in WEIGHTS}
    if groups[0].keys() != groups[4].keys():
        raise ValueError("Both configurations must contain matching games")
    zero_only = sum(groups[0][key] and not groups[4][key] for key in groups[0])
    four_only = sum(groups[4][key] and not groups[0][key] for key in groups[0])
    n = zero_only + four_only
    probability = (min(1.0, 2 * sum(comb(n, k) for k in range(min(zero_only, four_only) + 1)) / 2 ** n)
                   if n else 1.0)
    return {"matched_games_per_weight": len(groups[0]),
            "weight0_only_wins": zero_only, "weight4_only_wins": four_only,
            "win_rate_difference_0_minus_4": ((zero_only - four_only) / len(groups[0])
                                              if groups[0] else 0),
            "two_sided_exact_discordance_p": probability,
            "method": "Exact McNemar/binomial test on discordant outcomes, matched by seed and game index; matched initialization does not imply identical paths."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/borne_off_confirmation_100"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    files = ["evaluate_baseline.py", "confirm_borne_off.py", "validate_random_games.py",
             "monte_carlo/heuristic.py", "monte_carlo/rollout.py",
             "agents/monte_carlo_agent.py", "agents/gnubg_agent.py", "agents/random_agent.py",
             "backgammon/game.py", "backgammon/rules.py", "backgammon/state.py", "backgammon/move.py"]
    settings = {"weights": list(WEIGHTS), "seeds": list(SEEDS), "games_per_batch": 20,
                "rollouts": 20, "depth": 4, "max_turns": 10000,
                "opponent": "gnubg", "rollout_policy": "heuristic",
                "frozen_opponent": GNUBGAgent().benchmark_metadata(),
                "decision_rule": "Retain weight 4 unless 0 improves held-out win rate with two-sided exact matched-discordance p < 0.05; no automatic changes.",
                "candidate_defaults": asdict(DEFAULT_HEURISTIC),
                "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in files},
                "python_version": sys.version}
    manifest = args.output_dir / "manifest.json"
    if args.output_dir.exists():
        if not args.resume:
            parser.error("Output exists; use --resume to continue the same experiment")
        if not manifest.exists() or json.loads(manifest.read_text()) != settings:
            parser.error("Cannot resume: configuration, source, or Python changed")
    else:
        args.output_dir.mkdir(parents=True)
        atomic_json(manifest, settings)
    jobs = [(weight, seed, index, settings) for seed in SEEDS
            for index in range(20) for weight in WEIGHTS]

    def result_path(job):
        weight, seed, index, _ = job
        return args.output_dir / f"weight{weight}_seed{seed}_game{index + 1:02d}.json"

    done = []
    pending = []
    for job in jobs:
        path = result_path(job)
        if path.exists():
            result = json.loads(path.read_text())
            if (result["weight"], result["batch_seed"], result["index"]) != job[:3]:
                raise ValueError(f"Mismatched checkpoint {path}")
            done.append(result)
        else:
            pending.append(job)
    atomic_json(args.output_dir / "summary.json", results_summary(done, settings))
    print(f"Sweep: {len(done)}/200 already complete; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_game, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()  # Fail loudly; finished games remain checkpointed.
            atomic_json(result_path(job), result)
            done.append(result)
            atomic_json(args.output_dir / "summary.json", results_summary(done, settings))
            print(f"{len(done)}/200: weight={result['weight']}, seed={result['batch_seed']}, "
                  f"game={result['index'] + 1}, MC={'win' if result['game']['mc_won'] else 'loss'}, "
                  f"seconds={result['elapsed_seconds']:.1f}", flush=True)
    print(f"Complete: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
