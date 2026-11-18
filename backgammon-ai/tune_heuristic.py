"""Screen fixed heuristic candidates. This does not automatically select a winner."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from monte_carlo.heuristic import load_heuristic_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path, default=[
        Path("configs/heuristic_default.json"), Path("configs/heuristic_no_borne_off.json")])
    parser.add_argument("--seeds", nargs="+", type=int, default=[960000])
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--opponent", choices=("gnubg", "baseline", "heuristic"), default="gnubg")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("data/tuning_borne_off_screen"))
    args = parser.parse_args()
    if args.games < 2 or args.games % 2 or args.rollouts < 1 or args.depth < 0:
        parser.error("games must be even and >=2, rollouts positive, depth nonnegative")
    if len({p.stem for p in args.configs}) != len(args.configs):
        parser.error("configuration filenames must have unique stems")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("seed batches must be distinct")
    if any(abs(a - b) < 3 * args.games for a in args.seeds for b in args.seeds if a != b):
        parser.error("seed batches overlap: separate bases by at least 3 * games")
    try:
        for config in args.configs:
            load_heuristic_config(config)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new one to preserve experiments")
    args.output_dir.mkdir(parents=True)
    rows = []
    for config in args.configs:
        for seed in args.seeds:
            output = args.output_dir / f"{config.stem}_s{seed}.json"
            cmd = [sys.executable, "-B", str(Path(__file__).with_name("evaluate_baseline.py")),
                   "--opponent", args.opponent, "--games", str(args.games),
                   "--rollouts", str(args.rollouts), "--depth", str(args.depth),
                   "--rollout-policy", "heuristic", "--seed", str(seed),
                   "--heuristic-config", str(config), "--output", str(output)]
            print(f"Screening {config.name}, seed={seed}", flush=True)
            subprocess.run(cmd, check=True)
            report = json.loads(output.read_text())
            rows.append({"report": str(output), "settings": report["settings"],
                         "summary": report["summary"]})
            # Incremental ledger preserves finished batches if a later run fails.
            (args.output_dir / "screening_summary.json").write_text(json.dumps({
                "purpose": "exploratory screening, not held-out confirmation or automatic tuning",
                "comparison": "candidate parameters affect cutoff and both simulated players; actual opponent frozen",
                "runs": rows}, indent=2) + "\n")
    print(f"Screening complete: {args.output_dir}; do not select from small win-count differences alone.")


if __name__ == "__main__":
    main()
