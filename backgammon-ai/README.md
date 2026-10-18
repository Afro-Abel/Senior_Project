# Backgammon AI

Python prototype for AI-based playtesting of Backgammon rule variations.
Run these commands from this directory. Python 3.10 or newer is required;
the current code uses only the standard library.

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
