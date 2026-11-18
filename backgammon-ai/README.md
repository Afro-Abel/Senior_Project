# Backgammon AI

Python prototype for AI-based playtesting of Backgammon rule variations.
Run these commands from this directory. Python 3.10 or newer is required;
the current code uses only the standard library.

## Strategic styles: all four implemented

Select `--agent rear-priority --style-strength 0.05` to add an outcome-based
preference for progressing trailing checkers, or `--agent front-priority` to
favor progressing leading checkers. Use `--agent aggressive` to prefer actions
that hit more opposing checkers. The baseline cutoff heuristic and rollouts
stay shared. Use `--agent defensive` to prefer reducing direct blot threats
and building made points while contact remains.
See `agents/README.md` for formulas, checks, and pilot commands.
The evaluator saves separate MC/style scores and behavioral diagnostics.
Actual turn measurements now include per-agent hits, exposure, protected points,
and outcome-based advancement. `compare_styles.py` runs all five candidate types
on shared seed batches against the frozen baseline; see `agents/README.md`.

## Flat Monte Carlo baseline

`agents/monte_carlo_agent.py` evaluates every distinct legal resulting position using an equal
number of depth-limited rollouts. The opponent moves first after the
candidate. `depth` counts subsequent individual player turns, not dice, checker
steps, or pairs of turns. Forced passes consume a turn. Terminal positions stop
early and score +1 for a win or -1 for a loss from the decision maker's view.

At the cutoff, `monte_carlo/heuristic.py` supplies a shared utility. The raw score
is opponent pip count minus own pip count, plus 4 times the borne-off advantage.
Pip counts include 25 pips for each checker on the bar. While contact remains,
add 8 times the opponent-minus-own bar count, 2 times the home points made
advantage, 1.5 times the difference of squared longest prime lengths, and 3 times
the opponent-minus-own directly threatened blot count. Made points contain at
least two own checkers; primes are consecutive made points. Direct threats are
a single-die reachability proxy, including bar entry, not exact hit probabilities.
Combined-dice hits are not modeled by that feature.

The final nonterminal utility is `0.99 * tanh(raw / 50)`, so terminal outcomes
outrank any individual heuristic estimate. These are transparent starting weights,
not fitted or validated coefficients. Bar and borne-off terms intentionally
supplement their pip effects. Contact features are disabled once the players
have passed each other, leaving race progress to determine the score.

The agent averages utilities across rollouts and chooses the highest mean.
**This score is not a win percentage.** Equal scores are broken randomly, with
proven immediate wins preferred. There is no tree or style-specific bias yet;
all future styles should share this cutoff heuristic. Subsequent moves inside
rollouts use the selected policy (random by default). Candidate sequences reaching the same complete state
(board, bar counts, borne-off counts, and current player) are evaluated only
once, keeping the first legal sequence as the representative. The engine still
returns all legal sequences, and random rollout moves still sample those
sequences. For the default opening roll (2, 6) as P2, 28 sequences become 14
candidates, so 100 rollouts per candidate means 1,400 simulations.

Example engine integration:

```python
import random
from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.game import BackgammonGame

game = BackgammonGame(random.Random(42))
dice = game.roll_opening_dice()
agent = MonteCarloAgent(rollouts_per_move=10, depth=4, rng=random.Random(43))
move = agent.choose_move(game.state, dice)
if move is not None:
    game.apply_move(move)
if not game.is_game_over():
    game.switch_player()
```

A forced pass returns `None`; finished-game decisions raise `ValueError`.
The caller rolls the actual dice and manages actual turns. Simulations preserve
that game's state. The default depth is 4 turns, a configurable starting point
rather than a validated best horizon. Depth 0 evaluates each candidate directly
once, without sampling future turns.

Benchmark a single opening decision:

```sh
python3 -B benchmark_monte_carlo.py --rollouts 10 --depth 4 --seed 20261016
```

Reproducibility requires the same seed, code, and candidate ordering. Benchmark
runtime and heuristic correctness checks do not establish playing strength;
weights, depth, and simulation budgets still need pilot evaluation.

### Rollout policies

Both the benchmark and full-game evaluator accept `--rollout-policy random`
or `--rollout-policy heuristic`. The default stays random to preserve earlier
experiments. This option affects simulated moves for both players, not the
actual opponent. `--opponent heuristic` independently selects the actual
depth-zero opponent in complete games.

Under heuristic-guided rollouts, each simulated player chooses the legal unique
outcome with the highest immediate score from their own perspective, breaking
ties randomly. Dice remain random. There is no nested Monte Carlo search or
extra random exploration probability in this version. Both policies use the
same cutoff evaluation and depth definition; final rollout scores always use
the original decision maker's perspective. The heuristic weights are unchanged.
At depth zero, rollout policy has no effect.

For a controlled comparison, use separate reports with matching settings:

```sh
python3 -B evaluate_baseline.py --opponent heuristic --games 40 --rollouts 20 --depth 4 --rollout-policy random --seed 20261017 --output data/random_r20_d4_s20261017.json
python3 -B evaluate_baseline.py --opponent heuristic --games 40 --rollouts 20 --depth 4 --rollout-policy heuristic --seed 20261017 --output data/guided_r20_d4_s20261017.json
```

Guided rollouts evaluate many extra positions per simulated turn, so equal
rollout counts do not imply equal compute budgets. Compare timing as well as
win rate. Repeat with separate seed batches before drawing conclusions; changing
policy also changes random-number consumption, so paths are not identical.

## Optional external benchmark: gnubg-nn

The optional `GNUBGAgent` uses the pinned `gnubg-nn==1.1.0a11` evaluator.
Its adapter ranks our engine's distinct legal outcomes by estimated single-game
win probability at 0-ply evaluation. It ignores gammon/backgammon bonuses and
does not use the package's default `best_move()` ranking. This is a defined
policy built from the external evaluator, not a claim to reproduce the full
GNU Backgammon application's default playing strength. Do not compare its
0-ply setting directly with our rollout depth; they describe different work.

On Apple Silicon, use a separate Python 3.13 environment (the current system
Python 3.14 does not have a published wheel for this pinned package):

```sh
python3.13 -m venv .venv-gnubg
.venv-gnubg/bin/python -m pip install --only-binary=:all: -r requirements-gnubg.txt
.venv-gnubg/bin/python -B -m unittest discover -s tests -v
.venv-gnubg/bin/python -B evaluate_baseline.py --opponent gnubg --games 40 --rollouts 20 --depth 4 --rollout-policy heuristic --seed 940000 --output data/gnubg_r20_d4_s940000.json
```

The environment has already been created locally. It is excluded from Git.
`--opponent gnubg` defaults to `data/baseline_vs_gnubg.json` if no output path
is supplied. Reports include the pinned external version, objective, evaluation
depth, runtime platform, and hashes of the native module, weights, and data tables. Preserve output paths across
experiments. The random and heuristic opponents need no optional dependency.
Without it, only the seven external integration tests are skipped; conversion
tests still run. With the dependency installed, all 118 current tests run.

The board adapter stores each side in its own distance-to-off coordinates, with
bar checkers at slot 24 and borne-off counts implicit. Its probability evaluation
places the next player in row 1. Candidate scores therefore use one minus the
next player's win probability. Tests check a known starting Position ID, terminal
perspectives, both players' immediate wins, passes, input preservation, and
agreement between independent move generators in 378 position/player/dice
combinations. This sample is not exhaustive engine verification.

Use the external benchmark only under standard checker-play rules. It does not
understand future custom rule variations. Freeze its package and policy while
tuning, and reserve fresh seeds for final evaluation. If win rates are near zero,
use position-level external evaluations as an additional measure rather than
assuming all parameter settings are equally strong. The optional dependency is
GPL-3.0 licensed; see its license and upstream attribution when redistributing:
https://github.com/StonesAndDice/gnubg-nn-pypi

## Configurable heuristic parameters and initial tuning

Current numerical defaults are unchanged. Use a JSON file to override candidate
weights and scale; pip count stays fixed at weight 1. For example, the borne-off
ablation sets its extra bonus to zero while retaining the other features:

```sh
.venv-gnubg/bin/python -B evaluate_baseline.py --opponent gnubg --games 40 --rollouts 20 --depth 4 --rollout-policy heuristic --heuristic-config configs/heuristic_no_borne_off.json --seed 970000 --output data/no_borne_off_r20_d4_s970000.json
```

The candidate's configuration is used for both its cutoff scoring and the
heuristic choices inside its guided rollouts. Actual opponents remain frozen:
`heuristic` uses the original weights at depth zero, `baseline` uses the original
weights with 20 guided rollouts/depth 4, and `gnubg` uses the pinned external
0-ply win-probability policy. Reports save all resolved candidate parameters
and the frozen internal opponent settings. Benchmarking accepts the same
`--heuristic-config` flag. No tuning setting changes global defaults.

Run the initial baseline-versus-ablation screen:

```sh
.venv-gnubg/bin/python -B tune_heuristic.py --games 20 --seeds 960000 --output-dir data/tuning_borne_off_screen_s960000
```

This sequentially runs each configuration against the same frozen opponent and
seed batch. It preserves separate reports and updates a screening ledger after
each batch; an existing output directory is rejected to prevent overwrites.
Choose a new directory for additional runs. Defaults screen only the original
weights and the borne-off ablation; `--configs` accepts more files, `--seeds`
accepts multiple nonoverlapping batches, and `--opponent baseline` compares with
the frozen original MC. Small screens are exploratory and do not select a
winner automatically. See `configs/README.md` for the tuning/held-out protocol.

## Play in the terminal

```sh
python3 main.py
```

## Validate the engine

For full baseline-vs-random evaluation, run:

```sh
python3 -B evaluate_baseline.py --games 40 --rollouts 10 --depth 4 --seed 20261017 --output data/baseline_validation.json
```

This plays actual complete games, unlike the single-decision benchmark. Games
use independent seeded dice and agent streams. Monte Carlo alternates P1/P2
seats; the opening contest decides who starts. The game count must be even.
Every actual turn checks legal action selection, forced dice usage, state
preservation during decisions, checker conservation, and terminal winners.
A turn-limit violation raises an error rather than being counted as a draw.

The JSON report saves settings, source hashes, game seeds, per-game results,
and every Monte Carlo decision time. Summary timings include forced passes
and exclude the validation overhead. The approximate 95% Wilson interval
describes uncertainty against this random opponent, not human or expert strength.
The opponent samples legal move sequences, while MC candidates are deduplicated.
Use a separate output filename for each experiment to preserve previous runs.

For an exploratory heuristic-only comparison, use `--depth 0` with the same
settings and a different output file. Depth zero evaluates each candidate once;
the rollout count has no effect there. Reusing game seeds controls the dice
streams but does not make different agents visit identical positions. Comparing
two agents against random play does not establish their head-to-head ranking.
These experiments hold heuristic weights fixed; they do not tune them.

To compare lookahead directly against the same heuristic without lookahead:

```sh
python3 -B evaluate_baseline.py --opponent heuristic --games 40 --rollouts 10 --depth 4 --seed 20261017
```

`--opponent random` remains the default. The heuristic opponent uses depth zero,
one deterministic evaluation per unique candidate, and seeded random tie breaking.
It shares the exact same heuristic weights as the MC agent. `--rollouts` and
`--depth` configure the main MC agent only. Seat assignments still alternate.
This direct comparison can test whether sampled lookahead improves on the
cutoff evaluator alone, but one small batch is not conclusive.

Without `--output`, random matchups save to `data/baseline_validation.json`;
heuristic matchups save to `data/baseline_vs_heuristic.json`. Repeating the same
matchup overwrites its default report, so use explicit filenames for separate
settings/seeds. Reports identify `opponent`, `opponent_seed`, and
`opponent_wins` rather than labeling every opponent as random. Decision timing
still measures only the main MC agent.

Initial validation, seed `20261017`, 40 games per configuration:

- Depth 4 with 10 rollouts: 40 wins, 0 losses against random (20 wins in
  each seat); no validation failures. Mean decision time 0.0961 seconds,
  nearest-rank p95 0.3262 seconds, maximum 2.8640 seconds. Mean game length
  61.225 individual turns. Results: `data/baseline_validation.json`.
- Depth 0 heuristic-only: also 40 wins, 0 losses, with no validation failures.
  Results: `data/heuristic_only_validation.json`.
- Each 40/40 result has an approximate 95% Wilson interval of 91.2%–100%
  against this opponent. A perfect sample is not a guarantee of future wins.

These runs validate execution and show an advantage over random play; they
do not demonstrate that lookahead improves strength over heuristic-only play.
That requires direct comparison and further evaluation. Weights and budgets
remain provisional. Timings are local measurements, not performance guarantees.

```sh
python3 -B -m unittest discover -s tests -v
python3 -B test_current.py
python3 -B validate_random_games.py --games 100 --seed 20261016
```

The unittest fixtures check both players by mirroring positions. They cover
bar priority, blocked entry, hitting on entry, exact and oversized bearing off,
entering home and bearing off in one turn, doubles, and mandatory dice usage.
`test_current.py` contains the original prototype assertions.
Opening-roll tests cover both possible starters, repeated ties, seeded dice,
and the terminal game's use of the opening dice without an extra first-turn roll.

The random-game validator checks checker conservation and nonnegative bar/off
counts after each selected step, input-state preservation during move generation,
legal step sequences, maximum dice usage, the higher-die requirement, forced
passes, and termination with a winner. It searches dice usage separately from
the full-move generator, but shares the engine's single-step rules; the fixtures
provide separate expected outcomes for those rules. The turn limit raises an
error rather than silently treating an unfinished game as a draw. Failures
include the seed, game number, turn, player, and dice.

Dice and move choices use separate seeded random generators. The random agent
samples uniformly from legal move sequences; different sequences can reach the
same resulting board, so this is not uniform sampling of resulting positions.

This is engine validation, not a tournament or an estimate of playing strength.
Both the validator and terminal game use an opening contest: each player rolls
one die, ties are rerolled, and the higher roller starts using both opening dice.
Later turns use normal two-dice rolls, including doubles.
Engine callers should call `roll_opening_dice()` once before the first turn,
use its returned dice for that turn, then use `roll_dice()` on later turns.
Doubling-cube and match-scoring rules are also outside the current prototype.

Validation completed with seed `20261016`: 100/100 games finished, with no
validation failures using the opening contest. Games took 41–252 turns
(mean 103.6), with 3,522 hits, 3,522 bar entries, and 474 forced passes in total.

### Rule variation 1: at most one hit per turn

Standard rules remain the default. `--rule-variant single-hit` prohibits
a second hit during a complete turn, including bar-entry hits and doubles.
After hitting once, non-hitting moves remain legal. Maximum dice usage
and the higher-die requirement are computed under this restriction, rather
than filtering already-selected standard turns. The hit budget resets each
turn; pre-existing opponent checkers on the bar do not consume it.
The variant is carried by copied game states, so both actual players and
both players in MC rollouts use it. Heuristic weights and style strengths
remain fixed. The threatened-blot heuristic remains a single-die exposure
proxy, not an exact turn-level hit probability. GNU Backgammon evaluation
is restricted to standard rules. All four proposed variations are implemented as separate rule settings.

Validate the environment:

```bash
python3 -B validate_random_games.py --games 100 --rule-variant single-hit
```

Run a full-game evaluation (explicit output recommended):

```bash
python3 -B evaluate_baseline.py --agent aggressive --opponent baseline \
  --rule-variant single-hit --games 40 --rollouts 20 --depth 4 \
  --rollout-policy heuristic --seed 1402000 --output data/aggressive_single_hit.json
```

Omitting `--rule-variant` uses standard rules. The real game and simulations
share the variant; the baseline opponent keeps its frozen search settings.

### Rule variation 2: relaxed bar priority

`--rule-variant relaxed-bar` allows either bar entry or moving a board
checker while the player has checkers on the bar. Blocked bar entry does
not force a pass if board moves are available. Maximum dice usage and
the higher-die requirement still apply across all available actions.
Bearing off still requires an empty bar and all remaining checkers in
the standard six-point home board. This is a separate variation from
single-hit: multiple hits remain allowed.

The variant follows copied states into both players' rollouts. The
threatened-blot feature also considers opponent board attacks while the
opponent has checkers on the bar; feature weights are unchanged.

```bash
python3 -B validate_random_games.py --games 100 --rule-variant relaxed-bar
python3 -B evaluate_baseline.py --agent defensive --opponent baseline \
  --rule-variant relaxed-bar --games 40 --rollouts 20 --depth 4 \
  --rollout-policy heuristic --seed 1422000 --output data/defensive_relaxed_bar.json
```

### Rule variation 3: no doubles bonus

`--rule-variant no-doubles-bonus` treats doubles as two die uses rather
than four. A roll of (4, 4) can move one checker twice by four pips
(with a legal intermediate landing), or two checkers four pips each.
Both uses are mandatory when possible; otherwise use one, or pass if
neither is possible. Non-doubles, bar priority, hitting, and bearing off
retain standard rules. Variations are tested separately, not combined.
Copied rollout states retain the rule. The independent dice-usage checker
also expands doubles to two uses under this variant.

```bash
python3 -B validate_random_games.py --games 100 --rule-variant no-doubles-bonus
python3 -B evaluate_baseline.py --agent rear-priority --opponent baseline \
  --rule-variant no-doubles-bonus --games 40 --rollouts 20 --depth 4 \
  --rollout-policy heuristic --seed 1442000 --output data/rear_no_doubles_bonus.json
```

### Rule variation 4: eight-point bearing-off zone

`--rule-variant eight-point-bearoff` allows bearing off once all remaining
checkers are within eight points of the exit (P1 points 1--8; P2 points
17--24), with none on the bar. Dice still have values 1--6: a checker
seven or eight points from the exit must first move closer using a legal
intermediate landing. Exact removal remains allowed when other checkers
are farther back within the zone. Oversized removal remains forbidden
if any own checker is farther from the exit, including points seven and
eight. Maximum dice usage, higher-die priority, and standard doubles
remain in force. The variant is inherited by both players in rollouts.

This expands bearing-off eligibility only. The six entry points and
the heuristic's six-point home-board made-points feature stay unchanged;
all heuristic and style weights stay fixed.

```bash
python3 -B validate_random_games.py --games 100 --rule-variant eight-point-bearoff
python3 -B evaluate_baseline.py --agent front-priority --opponent baseline \
  --rule-variant eight-point-bearoff --games 40 --rollouts 20 --depth 4 \
  --rollout-policy heuristic --seed 1462000 --output data/front_eight_point_bearoff.json
```

Available rule settings: `standard`, `single-hit`, `relaxed-bar`,
`no-doubles-bonus`, and `eight-point-bearoff`. Select one at a time.

### Frozen-agent rule tournament pilot

`python3 -B run_tournament.py` runs all six pairings of rear-priority,
front-priority, aggressive, and defensive under standard rules and each
of the four variations separately: 30 matchups, 10 games each (300 total).
Each agent occupies each seat five times per matchup. Both agents use
20 heuristic-guided rollouts, depth 4, original heuristic weights, and
selected strengths 0.05, 0.01, 0.05, and 0.50 respectively. The opponent
is the other style, not the baseline.

Three worker processes run complete games concurrently. Every completed
game is saved under `data/rule_tournament_pilot`; the summary updates after
each completion. Resume an interrupted run with `--resume`; sources,
Python, and experiment settings must match. A directory lock prevents
duplicate writers. Workers are terminated if the runner encounters an error.
The manifest records settings, all source hashes, and the seed design.

Within each pair, variants share initial dice/agent seeds; game paths can
diverge. Distinct pair seed ranges do not overlap. All decisions are checked
for legality, maximal dice usage, state immutability, and checker conservation.
Full records include both sides' decision times, style metrics, and actual
turn behavior. Summary win rates identify agent A and B explicitly.

Rule activation measures turns where distinct legal resulting positions
differ from standard rules on the same visited state and dice. It measures
available-action changes, not whether the chosen move would be illegal
under standard rules or the causal effect on winning. Standard-rule games
have zero activation by definition. This pilot checks functionality,
runtime, behavior, and activation; 10 games per matchup do not establish
reliable agent rankings. Do not retune weights using these pilot outcomes.

```bash
nohup python3 -u -B run_tournament.py > data/rule_tournament_pilot.log 2>&1 &
tail -f data/rule_tournament_pilot.log
```
