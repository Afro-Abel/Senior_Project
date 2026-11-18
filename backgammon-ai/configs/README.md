# Heuristic tuning configurations

All parameters are candidate-agent settings, used for cutoff evaluation and for
both players' simulated choices inside that candidate's guided rollouts.
The actual opponent's settings are independent and remain frozen.

- `borne_off`: extra reward per checker borne off (default 4).
- `bar`: extra opponent-minus-own bar reward (8).
- `home_points`: made home-point advantage (2).
- `prime_squared`: difference of squared longest prime lengths (1.5).
- `threatened_blots`: opponent-minus-own directly threatened blot count (3).
- `scale`: positive divisor before tanh normalization (50).

Pip-count weight stays fixed at 1. Contact-only feature behavior and terminal
scores are unchanged. Coefficients must be finite and nonnegative; scale must
be strictly positive. Partial JSON files inherit unspecified defaults. Unknown
keys are errors, avoiding silent misspelled parameters.

## Initial screening plan

Compare the original heuristic with an ablation of the borne-off bonus alone:
`heuristic_default.json` versus `heuristic_no_borne_off.json`. Keep 20 guided
rollouts and depth 4 fixed. The first exploratory batch uses base seed 960000,
20 games per configuration, and equal player assignments. It is too small for
fine ranking or a claim of improvement. Retain the original weights unless
further evidence supports a change.

Next, screen other feature ablations independently, then try a few nearby values
for useful terms. Compare promising configurations with the frozen original MC
(`--opponent baseline`) as well as the fixed external evaluator (`gnubg`). Test
combined changes afterward because features interact. Tune normalization scale
separately from the relative feature weights. Neither a parameter grid nor
hill climbing automatically establishes optimality.

Use fresh, nonoverlapping seed ranges for follow-up screening. Reserve base
seed 1000000 and its required game streams for final held-out confirmation;
do not inspect it repeatedly while selecting parameters. Reusing a base seed
across configurations matches RNG initialization, not every simulated path.
Policies diverge and consume randomness differently.

All results must retain resolved weights, search settings, opponent settings,
source hashes, and external engine metadata. Compare both wins and runtime,
and report uncertainty. Freeze selected parameters before style/rule experiments.

## First screening result

The initial seed-960000 screen completed without validation failures. Original
weights won 3/20 games against the external reference; removing the borne-off
bonus won 1/20. Mean decision times were approximately 0.492 and 0.493 seconds.
This small sample does not justify changing the default or establish a causal
benefit for weight 4. Keep the defaults for now and use additional screening
and held-out confirmation before selecting parameters. Both reports and the
incremental ledger are preserved in
`data/tuning_borne_off_screen_s960000/`.

## Borne-off weight sweep: 100 games per value

`sweep_borne_off.py` evaluates weights 0, 2, 4, 6, and 8 against the frozen
original Monte Carlo agent. Both sides use 20 guided rollouts and depth 4.
Only the candidate's borne-off weight changes. Five fresh seed batches
(980100, 980200, 980300, 980400, 980500) supply 20 games each, with ten games
per seat per batch: 100 games per weight and 500 games total. Seed 1000000
remains reserved for held-out confirmation. The five explicit JSON files
`heuristic_borne_off_0.json` through `heuristic_borne_off_8.json` record the
same candidate configurations; the runner snapshots their effective values
from the frozen defaults and its selected weights in `manifest.json`.

The timing pilot used two games at seed 980000, separate from sweep data.
It measured approximately 31.2 seconds per complete game, suggesting about
4.3 hours sequentially or 1.4 hours at ideal three-worker scaling. Actual time
may be longer because of contention, thermal limits, and game variability.
The pilot's two wins are not evidence that weight 4 beats an identical agent.

Start a foreground sweep from `backgammon-ai`:

```sh
python3 -B sweep_borne_off.py --workers 3 --output-dir data/borne_off_sweep_100
```

A background run was started during setup. Its process ID is recorded in
`data/borne_off_sweep_100.pid`; follow its log with:

```sh
tail -f data/borne_off_sweep_100.log
```

Each completed game is saved atomically as a separate checkpoint.
`data/borne_off_sweep_100/summary.json` shows completed/expected game counts
and per-weight summaries; `status` becomes `complete` only at 500 games.
Do not interpret partial rankings. After a stopped run, resume with:

```sh
python3 -B sweep_borne_off.py --workers 3 --output-dir data/borne_off_sweep_100 --resume
```

Do not start a second run against the same directory while one is active.
Resume requires matching Python, code hashes, and experiment settings. Keep
the machine awake and avoid editing experiment source files during the run.
Changing worker count does not change the per-game RNG seeds.

The sweep does not update defaults or choose a global optimum. Once complete,
inspect variation across batches, uncertainty, and runtime, then test promising
nearby weights if warranted. Check a finalist against the external reference
and fresh held-out games before adopting it.

## Held-out confirmation of weights 0 and 4

The completed 500-game tuning sweep produced 53/100 wins for weight 0, 44/100
for 2, 50/100 for 4, 50/100 for 6, and 49/100 for 8 against the frozen original
MC. No default changed. Weight 0 was selected only as a candidate for confirmation.

`confirm_borne_off.py` now compares weights 0 and 4 against the pinned external
gnubg-nn opponent, with 20 guided rollouts and depth 4. It uses 20 games each
at base seeds 1000000, 1000100, 1000200, 1000300, and 1000400: 100 games per
weight, 200 total, with equal seats. These seed ranges were checked against
saved gameplay records before launch and are now allocated to this single
confirmation comparison. Do not reuse them as held-out data for later tuning.

Before the run, the rule was set to retain weight 4 unless weight 0 has a higher
win rate and a two-sided exact matched-discordance (McNemar/binomial) p-value
below 0.05. Matching uses game index and initialized RNG streams; different
policies may still produce different trajectories. The result is not proof of
a globally optimal weight, and the runner never changes constants automatically.

Three workers run in the background. Monitor:

```sh
tail -f data/borne_off_confirmation_100.log
```

Per-game checkpoints, pinned external model/data hashes, source hashes, and
settings are saved under `data/borne_off_confirmation_100/`. The summary's
matched comparison is computed only after all 200 games finish. Do not select
from partial results. If a stopped run needs resuming, use the same environment:

```sh
.venv-gnubg/bin/python -B confirm_borne_off.py --workers 3 --output-dir data/borne_off_confirmation_100 --resume
```

Do not launch a second writer to an active directory. Resume refuses changed
source, interpreter, or external benchmark metadata. The earlier sweep data
remain untouched.
