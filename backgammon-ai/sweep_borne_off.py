"""Resumable borne-off sweep: five weights, five batches, 100 games per weight."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from evaluate_baseline import play_game, summarize
from monte_carlo.heuristic import DEFAULT_HEURISTIC, HeuristicConfig


WEIGHTS = (0, 2, 4, 6, 8)
SEEDS = (980100, 980200, 980300, 980400, 980500)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_game(job):
    weight, seed, index, settings = job
    start = perf_counter()
    config = replace(DEFAULT_HEURISTIC, borne_off=weight)
    result = play_game(index, seed, settings["rollouts"], settings["depth"],
                       settings["max_turns"], "baseline", "heuristic", config)
    return {"weight": weight, "batch_seed": seed, "index": index,
            "heuristic_config": asdict(config), "game": result,
            "elapsed_seconds": perf_counter() - start}


def results_summary(results, settings):
    groups = []
    for weight in settings["weights"]:
        selected = [r for r in results if r["weight"] == weight]
        if selected:
            groups.append({"borne_off": weight, "completed_games": len(selected),
                           "summary": summarize([r["game"] for r in selected], "baseline"),
                           "game_elapsed_seconds": sum(r["elapsed_seconds"] for r in selected)})
    expected = len(settings["weights"]) * len(settings["seeds"]) * settings["games_per_batch"]
    return {"status": "complete" if len(results) == expected else "running",
            "completed_games": len(results), "expected_games": expected,
            "note": "Tuning data only. Do not select weights from partial results; held-out confirmation remains necessary.",
            "candidates": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("data/borne_off_sweep_100"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(__file__).resolve().parent
    files = ["evaluate_baseline.py", "sweep_borne_off.py", "validate_random_games.py",
             "monte_carlo/heuristic.py", "monte_carlo/rollout.py",
             "agents/monte_carlo_agent.py", "agents/gnubg_agent.py", "agents/random_agent.py",
             "backgammon/game.py", "backgammon/rules.py", "backgammon/state.py", "backgammon/move.py"]
    settings = {"weights": list(WEIGHTS), "seeds": list(SEEDS), "games_per_batch": 20,
                "rollouts": 20, "depth": 4, "max_turns": 10000,
                "opponent": "baseline", "rollout_policy": "heuristic",
                "frozen_opponent": {"config": asdict(DEFAULT_HEURISTIC), "rollouts": 20,
                                    "depth": 4, "rollout_policy": "heuristic"},
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
    print(f"Sweep: {len(done)}/500 already complete; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_game, job): job for job in pending}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()  # Fail loudly; finished games remain checkpointed.
            atomic_json(result_path(job), result)
            done.append(result)
            atomic_json(args.output_dir / "summary.json", results_summary(done, settings))
            print(f"{len(done)}/500: weight={result['weight']}, seed={result['batch_seed']}, "
                  f"game={result['index'] + 1}, MC={'win' if result['game']['mc_won'] else 'loss'}, "
                  f"seconds={result['elapsed_seconds']:.1f}", flush=True)
    print(f"Complete: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
