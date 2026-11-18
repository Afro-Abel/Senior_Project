"""Resumable style comparison against the fixed original MC, with shared seeds."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

from evaluate_baseline import play_game, summarize
from monte_carlo.heuristic import DEFAULT_HEURISTIC


AGENTS = ("baseline", "rear-priority", "front-priority", "aggressive", "defensive")


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run_game(job):
    agent, seed, index, settings = job
    start = perf_counter()
    record = play_game(index, seed, 20, 4, 10000, "baseline", "heuristic",
                       DEFAULT_HEURISTIC, agent, settings["style_strength"])
    return {"agent": agent, "batch_seed": seed, "index": index, "game": record,
            "elapsed_seconds": perf_counter() - start}


def make_summary(results, settings):
    expected = len(AGENTS) * len(settings["seeds"]) * settings["games_per_batch"]
    groups = []
    for agent in AGENTS:
        selected = [r["game"] for r in results if r["agent"] == agent]
        if selected:
            groups.append({"agent": agent, "summary": summarize(selected, "baseline")})
    return {"status": "complete" if len(results) == expected else "running",
            "completed_games": len(results), "expected_games": expected,
            "notes": ["Exploratory behavior validation, not final rule-variation results.",
                      "All candidates face the same baseline; this is not an all-pairs tournament.",
                      "Shared RNG initialization does not imply identical game paths.",
                      "Use opportunity-conditioned and contact-phase measures alongside raw totals.",
                      "Do not infer style or strength from small samples or partial rankings."],
            "agents": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--games-per-batch", type=int, default=20)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1121000, 1122000])
    parser.add_argument("--output-dir", type=Path, default=Path("data/style_behavior_comparison"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.games_per_batch < 2 or args.games_per_batch % 2:
        parser.error("workers must be positive; games-per-batch must be even and >=2")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("seeds must be distinct nonnegative integers")
    if any(abs(a - b) < 3 * args.games_per_batch for a in args.seeds for b in args.seeds if a != b):
        parser.error("seed batches overlap")
    root = Path(__file__).resolve().parent
    files = [Path(name) for name in ("compare_styles.py", "evaluate_baseline.py", "behavior_metrics.py", "validate_random_games.py")]
    files += [p.relative_to(root) for folder in ("agents", "backgammon", "monte_carlo")
              for p in sorted((root / folder).glob("*.py"))]
    settings = {"agents": list(AGENTS), "seeds": args.seeds,
                "games_per_batch": args.games_per_batch, "style_strength": .05,
                "rollouts": 20, "depth": 4, "rollout_policy": "heuristic", "opponent": "baseline",
                "heuristic_config": asdict(DEFAULT_HEURISTIC), "python_version": sys.version,
                "source_sha256": {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
    manifest = args.output_dir / "manifest.json"
    if args.output_dir.exists():
        if not args.resume or not manifest.exists() or json.loads(manifest.read_text()) != settings:
            parser.error("Output exists: resume only with matching settings, source, and Python")
    else:
        args.output_dir.mkdir(parents=True)
        atomic_json(manifest, settings)
    jobs = [(agent, seed, index, settings) for seed in args.seeds
            for index in range(args.games_per_batch) for agent in AGENTS]
    def path_for(job):
        agent, seed, index, _ = job
        return args.output_dir / f"{agent}_seed{seed}_game{index + 1:03d}.json"
    results, pending = [], []
    for job in jobs:
        path = path_for(job)
        if path.exists():
            result = json.loads(path.read_text())
            if (result["agent"], result["batch_seed"], result["index"]) != job[:3]:
                raise ValueError(f"Invalid checkpoint {path}")
            results.append(result)
        else:
            pending.append(job)
    atomic_json(args.output_dir / "summary.json", make_summary(results, settings))
    print(f"Style comparison: {len(results)}/{len(jobs)} complete; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_game, job): job for job in pending}
        for future in as_completed(futures):
            result = future.result()
            atomic_json(path_for(futures[future]), result)
            results.append(result)
            atomic_json(args.output_dir / "summary.json", make_summary(results, settings))
            print(f"{len(results)}/{len(jobs)}: {result['agent']}, seed={result['batch_seed']}, "
                  f"game={result['index']+1}, win={result['game']['mc_won']}", flush=True)
    print(f"Complete: {args.output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
