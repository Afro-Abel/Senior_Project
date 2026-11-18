"""Resumable all-pairs pilot of frozen styles under five separate rule settings."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import fcntl
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter

from agents.style_defaults import SELECTED_STYLE_STRENGTHS
from backgammon.variants import RULE_VARIANTS
from behavior_metrics import summarize_turns
from evaluate_baseline import play_game, wilson_interval
from monte_carlo.heuristic import DEFAULT_HEURISTIC

PAIRS = tuple(combinations(SELECTED_STYLE_STRENGTHS, 2))


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def jobs_for(settings):
    # Share initialization across variants for a given pair/game. Paths diverge.
    return [(variant, a, b, index, settings['seed'] + pair_index * 1000)
            for index in range(settings['games_per_matchup'])
            for variant in settings['variants']
            for pair_index, (a, b) in enumerate(PAIRS)]


def run_game(job):
    variant, a, b, index, seed = job
    start = perf_counter()
    game = play_game(index, seed, 20, 4, 10000, opponent_kind=b,
                     rollout_policy='heuristic', heuristic_config=DEFAULT_HEURISTIC,
                     agent_kind=a, style_strength=SELECTED_STYLE_STRENGTHS[a],
                     rule_variant=variant, collect_rule_impact=True)
    return {'variant': variant, 'agent_a': a, 'agent_b': b, 'index': index,
            'batch_seed': seed, 'game': game, 'elapsed_seconds': perf_counter() - start}


def summarize_results(results, settings):
    matchups = []
    standings = []
    for variant in settings['variants']:
        for a, b in PAIRS:
            rows = [r for r in results if (r['variant'], r['agent_a'], r['agent_b']) == (variant, a, b)]
            if not rows:
                continue
            games = [r['game'] for r in rows]
            wins = sum(g['mc_won'] for g in games)
            checked = sum(g['rule_impact']['turns_checked'] for g in games)
            changed = sum(g['rule_impact']['turns_with_changed_outcomes'] for g in games)
            matchups.append({'variant': variant, 'agent_a': a, 'agent_b': b,
                'games': len(games), 'a_wins': wins, 'b_wins': len(games)-wins,
                'a_win_rate': wins / len(games),
                'a_win_rate_95pct_wilson': wilson_interval(wins, len(games)),
                'a_by_seat': {str(p): {'games': sum(g['mc_player'] == p for g in games),
                    'wins': sum(g['mc_player'] == p and g['mc_won'] for g in games)} for p in (1, -1)},
                'mean_turns': mean(g['turns'] for g in games),
                'rule_impact': {'turns_checked': checked, 'turns_with_changed_outcomes': changed,
                                'changed_outcome_turn_rate': changed / checked},
                'mean_worker_game_seconds': mean(r['elapsed_seconds'] for r in rows)})
        for agent in SELECTED_STYLE_STRENGTHS:
            rows = [r for r in results if r['variant'] == variant and agent in (r['agent_a'], r['agent_b'])]
            if not rows:
                continue
            wins = sum(r['game']['mc_won'] if r['agent_a'] == agent else not r['game']['mc_won'] for r in rows)
            turns, times, styles = [], [], []
            for row in rows:
                g = row['game']; role = 'candidate' if row['agent_a'] == agent else 'opponent'
                turns.extend(t for t in g['behavior_turns'] if t['role'] == role)
                times.extend(g['decision_seconds'] if role == 'candidate' else g['opponent_decision_seconds'])
                styles.extend(g['style_decisions'] if role == 'candidate' else g['opponent_style_decisions'])
            standings.append({'variant': variant, 'agent': agent, 'games': len(rows), 'wins': wins,
                'win_rate': wins / len(rows), 'gameplay_behavior': summarize_turns(turns),
                'mean_decision_seconds': mean(times),
                'style_overrode_mc_decisions': sum(s['style_overrode_mc'] for s in styles),
                'mean_mc_utility_sacrificed': mean(s['mc_utility_sacrificed'] for s in styles) if styles else None})
    expected = len(PAIRS) * len(settings['variants']) * settings['games_per_matchup']
    return {'status': 'complete' if len(results) == expected else 'running',
            'completed_games': len(results), 'expected_games': expected,
            'notes': ['Pilot for runner verification, runtime, and rule activation; not final strength rankings.',
                      'Ten games per matchup have large uncertainty; do not select weights from these results.',
                      'Rule impact compares distinct available outcomes against standard rules on visited states.',
                      'Shared initial seeds do not imply identical game paths. Turn observations are not independent samples.',
                      'Mean worker game time includes contention and validation; it is not serial engine speed.'],
            'matchups': matchups, 'standings': standings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--games-per-matchup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1501000)
    parser.add_argument('--variants', choices=RULE_VARIANTS, nargs='+', default=list(RULE_VARIANTS))
    parser.add_argument('--output-dir', type=Path, default=Path('data/rule_tournament_pilot'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or not 2 <= args.games_per_matchup <= 332 or args.games_per_matchup % 2:
        parser.error('workers must be positive; games-per-matchup must be even and between 2 and 332')
    if args.seed < 0 or len(set(args.variants)) != len(args.variants):
        parser.error('seed must be nonnegative and variants must be distinct')
    root = Path(__file__).resolve().parent
    files = [Path(n) for n in ('run_tournament.py', 'evaluate_baseline.py', 'behavior_metrics.py',
                               'rule_impact.py', 'validate_random_games.py')]
    files += [p.relative_to(root) for folder in ('agents', 'backgammon', 'monte_carlo')
              for p in sorted((root / folder).glob('*.py'))]
    settings = {'schema_version': 1, 'variants': args.variants, 'pairs': [list(p) for p in PAIRS],
        'games_per_matchup': args.games_per_matchup, 'seed': args.seed,
        'style_strengths': dict(SELECTED_STYLE_STRENGTHS),
        'rollouts': 20, 'depth': 4, 'rollout_policy': 'heuristic', 'max_turns': 10000,
        'heuristic_config': asdict(DEFAULT_HEURISTIC), 'python_version': sys.version,
        'source_sha256': {str(p): hashlib.sha256((root / p).read_bytes()).hexdigest() for p in files}}
    output = args.output_dir; manifest = output / 'manifest.json'
    if output.exists():
        if not args.resume or not manifest.exists() or json.loads(manifest.read_text()) != settings:
            parser.error('Output exists: resume requires matching settings, sources, and Python')
    else:
        output.mkdir(parents=True)
        atomic_json(manifest, settings)
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another process is writing this tournament')
        def path_for(job):
            v, a, b, i, seed = job
            return output / f'{v}_{a}_vs_{b}_seed{seed}_game{i+1:03d}.json'
        results, pending = [], []
        for job in jobs_for(settings):
            path = path_for(job)
            if not path.exists():
                pending.append(job); continue
            result = json.loads(path.read_text())
            if tuple(result[k] for k in ('variant','agent_a','agent_b','index','batch_seed')) != job:
                raise ValueError(f'Invalid checkpoint {path}')
            results.append(result)
        total = len(results) + len(pending)
        atomic_json(output / 'summary.json', summarize_results(results, settings))
        print(f'Tournament pilot: {len(results)}/{total} complete; workers={args.workers}', flush=True)
        started = perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_game, job): job for job in pending}
            try:
                for future in as_completed(futures):
                    result = future.result()
                    atomic_json(path_for(futures[future]), result)
                    results.append(result)
                    atomic_json(output / 'summary.json', summarize_results(results, settings))
                    print(f"{len(results)}/{total}: {result['variant']}, {result['agent_a']} vs {result['agent_b']}, game={result['index']+1}", flush=True)
            except BaseException:
                pool.terminate_workers()
                raise
        atomic_json(output / 'timing.json', {'resumed_session_wall_seconds': perf_counter()-started,
                                            'games_completed_this_session': len(pending)})
        print(f'Complete: {output / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
