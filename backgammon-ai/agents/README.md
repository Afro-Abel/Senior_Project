# Strategic agents

All four planned styles are implemented: `RearPriorityAgent`,
`FrontPriorityAgent`, `AggressiveAgent`, and `DefensiveAgent`. Their behavioral
differences and relative strength still require systematic validation.

## Rear-priority definition

For each player, define `F(state) = sum(checker_count * distance_to_off**2)`.
Distances are 1–24 on the board, 25 on the bar, and 0 after bearing off. The
raw preference for a complete action is `F(before) - F(after)`. For example,
advancing a checker from distance 20 to 18 earns 76, while advancing from 8
to 6 earns 28. This favors progressing trailing checkers; it does not force
the farthest checker to move regardless of the Monte Carlo evaluation.

This definition is based on complete outcomes, not the order of steps or the
identity of checkers. Equivalent resulting states get equal raw preference.
Paths that hit different opponents remain distinct even if their own checkers
finish on the same points. Their style preferences may tie; MC scores still
account for the differing opposing positions.

For each decision, min-max normalize raw preferences across distinct candidates
to [0, 1]. If all preferences are equal, all bonuses are zero. The selection
score is `mean_MC_utility + style_strength * normalized_preference`. The initial
strength is 0.05, a provisional parameter that still needs behavioral tuning.
It limits the per-candidate bonus to 0.05. A proven immediate win always takes
precedence, even when another candidate's adjusted score is above 1.

The raw MC utility is unchanged. Rollout moves, cutoff weights, normalization
scale, depth, and simulation count are shared with the baseline. Style bonuses
are applied only when choosing the actual candidate, not inside rollouts.
At zero strength the same seed produces the baseline's scores, tie behavior,
and RNG consumption. Direct construction defaults to the provisional baseline:
20 guided rollouts, depth 4. The evaluator's explicit CLI settings override
these defaults, so use the complete command below for research comparisons.

## Running a rear-priority pilot

From `backgammon-ai`:

```sh
python3 -B evaluate_baseline.py --agent rear-priority --opponent baseline --games 10 --rollouts 20 --depth 4 --rollout-policy heuristic --style-strength 0.05 --seed 1101000 --output data/rear_priority_vs_baseline_s1101000.json
```

`--agent baseline` remains the default; the opponent does not inherit the
candidate's style. The benchmark also accepts `--agent rear-priority` and
`--style-strength`, and prints MC utility separately from the added bonus.

Reports contain style definitions/settings and per-decision diagnostics:
candidate count, range of raw preferences, chosen normalized preference,
bonus, raw MC utility, adjusted score, and MC utility sacrificed. The summary
counts opportunities with different preferences and decisions in which the
bias displaced a strictly higher MC estimate. A style-resolved MC tie is not
counted as sacrificing utility. These are decision-level diagnostics, not proof
that the agent has superior playing strength or a globally distinct style.

The initial two-game integration pilot against heuristic-only play used 20
guided rollouts, depth 4, strength 0.05, and seed 1100000. Both games finished
without validation failures (one win and one loss). In 10 decisions the agent
selected a candidate with a lower MC estimate because of the bias. This verifies
that the preference can affect play; it is too small a sample for performance
or behavioral conclusions. Results: `data/rear_priority_pilot_s1100000.json`.

Before moving to formal tournaments, compare matched baseline and rear-priority
runs across fresh seed batches, examine actual checker progression as well as
these diagnostics, and tune strength to produce a detectable but modest bias.

## Front-priority definition

`FrontPriorityAgent` shares the wrapper, cutoff heuristic, rollout policy,
normalization, and immediate-win guard. Its raw preference is the decrease in
`G(state) = sum(checker_count * (52 * distance - distance**2))`. Distances are
1–24 on board, 25 on the bar, and 0 off. Equivalently, its raw score is
`52 * pip_progress - rear_raw_score`.

For the same two-pip advance, moving from distance 8 to 6 scores 76, while
moving from 20 to 18 scores 28: the reverse of rear priority. The coefficient
52 is twice 26 (one above the maximum distance 25), keeping the potential
increasing throughout the distance range while rewarding progress more heavily
near the exit. It belongs to this style definition, not the baseline heuristic.
Unlike simply negating rear priority, it preserves a positive reward for progress
when actual pip advances differ.

When all candidates advance the same number of pips and preferences differ,
normalized front and rear preferences are complements. This need not hold
when actual advances differ, such as oversized bearing-off rolls. Complete
outcome scoring ensures equivalent sequences always receive equal preferences.
The provisional bias strength is the same 0.05 used for rear priority.

Run a pilot against the frozen baseline:

```sh
python3 -B evaluate_baseline.py --agent front-priority --opponent baseline --games 10 --rollouts 20 --depth 4 --rollout-policy heuristic --style-strength 0.05 --seed 1103000 --output data/front_priority_vs_baseline_s1103000.json
```

Both the evaluator and benchmark support this agent. Reports identify its
formula and behavioral diagnostics separately. The initial two-game integration
pilot against heuristic-only play completed without validation failures (one
win and one loss). The bias displaced a higher MC estimate in six decisions.
This verifies operation, not playing strength or a globally distinct style.
The report is `data/front_priority_pilot_s1102000.json`.

## Aggressive definition

`AggressiveAgent` rewards the number of opposing checkers hit during the
candidate's complete turn: `opponent_bar_after - opponent_bar_before`. Existing
bar checkers do not earn credit again. The opponent cannot re-enter during our
turn, so this difference counts the hits exactly, including hits on bar entry
and multiple hits by the same checker. No heuristic weights are changed.

The raw hit count is min-max normalized across distinct candidate outcomes,
then multiplied by the same provisional strength 0.05. If no hits are possible,
or every outcome hits the same number, there is no bonus. The unchanged MC
utility can outweigh the preference, and immediate wins always take priority.
Different paths ending with the same own-checker placement remain distinct
if their hits produce different opposing board/bar states.

Run a pilot against the frozen baseline:

```sh
python3 -B evaluate_baseline.py --agent aggressive --opponent baseline --games 10 --rollouts 20 --depth 4 --rollout-policy heuristic --style-strength 0.05 --seed 1105000 --output data/aggressive_vs_baseline_s1105000.json
```

This uses the same cutoff evaluation, guided continuation policy, depth,
rollout budget, and style normalization as the front/rear agents. The evaluator
and benchmark both label the agent as aggressive and save its style settings
and decision-level diagnostics. Higher normalized preferences indicate a bias
toward more immediate hits among that turn's choices; broad behavioral claims
still require comparison with the baseline across complete games and seeds.

The two-game integration pilot at seed 1104000 completed without validation
failures (one win, one loss). Of 16 decisions with different hitting preferences,
the bias displaced a strictly higher MC estimate four times. This confirms
operation, not an established advantage or higher hitting rate over baseline.
Results: `data/aggressive_pilot_s1104000.json`.

## Defensive definition

`DefensiveAgent` uses this raw candidate preference:

```
(threatened_own_blots_before - threatened_own_blots_after)
+ (made_own_points_after - made_own_points_before)  # only if contact remains after
```

The two terms initially have equal weight 1. A made point has at least two own
checkers and can be anywhere, including an anchor in the opponent's home board.
Closing two threatened blots into one protected point therefore earns 3: two
threats removed and one point made. Breaking a point and creating two threatened
blots scores -3. A negative score is allowed; normalization favors the least
harmful option if every candidate makes safety worse.

Threats reuse the shared direct single-die proxy, including opposing bar entry.
This is not an exact probability and omits combined-dice hits. When the opponent
has bar checkers, the proxy considers their direct entry threats rather than
other board-checker movements after re-entry. It can therefore miss some threats.

Made-point changes are ignored once the players have passed each other and no
checkers remain on the bar. This prevents a bonus for irrelevant blocking in a
pure race. The change is gated based on the resulting position, rather than
subtracting two gated safety potentials, so entering a race is not penalized
just because previously useful protected points lose their relevance. Removing
actual direct threats can still earn credit when an action exits contact.

As with the other styles, min-max normalization and the provisional 0.05
strength affect only actual candidate selection; cutoff evaluation, simulated
policies, and budgets stay shared. Zero bias reproduces the baseline. Proven
immediate wins are always chosen ahead of nonterminal style preferences.

Run the defensive pilot against the frozen baseline:

```sh
python3 -B evaluate_baseline.py --agent defensive --opponent baseline --games 10 --rollouts 20 --depth 4 --rollout-policy heuristic --style-strength 0.05 --seed 1107000 --output data/defensive_vs_baseline_s1107000.json
```

Both the evaluator and benchmark accept `--agent defensive`, save the defensive
formula/settings, and record the same decision diagnostics used for the other
styles. Equal term weights and strength 0.05 are starting choices, not tuned
values. The next research step is comparing all four styles and the baseline
on shared seed batches using actual per-agent hits, blot exposure, protected
points, and checker advancement, rather than inferring style from win rates
or the biased agents' own normalized preferences alone.

The initial two-game integration pilot against heuristic-only play at seed
1106000 finished with two wins and no validation failures. The bias displaced
a higher MC estimate in five decisions. These results verify operation, not
superiority or a distinct defensive style over full experiments. Report:
`data/defensive_pilot_s1106000.json`.

## Comparing measured behavior across all styles

`behavior_metrics.py` records actual turns for both the candidate and opponent.
It is independent of the style bonus and does not measure simulated rollout
positions. Old reports without `behavior_turns` cannot recover these statistics.

- Hits and hits per turn are attributed to the player performing the action.
- Hitting opportunities count turns with at least one legal complete action
  that hits. `hit_when_available_rate` conditions on those opportunities.
- Blots, directly threatened blots, and made points are sampled after each
  player's own turn, including forced passes. Threats use the same approximate
  single-die proxy as the baseline; they are not exact hitting probabilities.
- Contact-only exposure and made-point means exclude resulting pure-race states,
  reducing distortion from differing amounts of race play. The denominator is
  also saved; a missing phase or opportunity produces null, not a fabricated zero.
- Advancement midpoint is the reduction in own squared distances divided by
  twice own pip progress. A two-pip move from distance 20 to 18 has midpoint 19,
  versus 7 for a move from 8 to 6. It is outcome-based, not dependent on which
  equivalent step sequence represents the action. Aggregate midpoint is weighted
  by pips advanced, not an unweighted average of per-turn means.
- `advancement_choice_rank` min-max ranks that midpoint among the current legal
  outcomes. Lower means relatively front-focused, higher rear-focused. Turns
  with no variation are excluded from its mean and counted separately.

Do not equate each style's own normalized bonus with proof of distinct behavior.
Inspect these common measures, stage/opportunity counts, win rates, and variation
across seed batches together. Hits, exposure, and checker distribution may affect
one another; the comparison is exploratory rather than proof of a unique cause.

The shared-seed comparison uses all five candidate types against the same frozen
original MC opponent. It is not an all-pairs tournament. Search is fixed at 20
guided rollouts and depth 4, and each biased agent uses strength 0.05. The baseline
candidate has no style bonus. Two default batches (1121000 and 1122000) each supply
20 games per candidate, balanced across sides: 40 games per candidate, 200 total.

```sh
python3 -B compare_styles.py --workers 3 --output-dir data/style_behavior_comparison
```

The runner checkpoints every completed game and writes a live summary. If stopped,
resume with the same code, Python, and settings:

```sh
python3 -B compare_styles.py --workers 3 --output-dir data/style_behavior_comparison --resume
```

Never run two writers in the same output directory. Defaults and existing
experiment data are not changed. The manifest records source hashes and settings.
Use new seed batches for final evaluation after choosing bias strengths. The
small integration smoke test uses two games per agent at seed 1120000; its results
are for checking execution and metric collection, not ranking styles.

The full 200-game comparison was launched in the background after the ten-game
smoke test completed without validation failures. All 118 current tests pass
when the optional external dependency is installed. Monitor the active run:

```sh
tail -f data/style_behavior_comparison.log
```

`data/style_behavior_comparison/summary.json` records progress and aggregates.
The smoke test averaged about 28 seconds per game under three workers, suggesting
roughly 31 minutes for 200 games at similar throughput; actual time can vary.
Keep the machine awake and avoid changing experiment source while it runs.
Terminal positions are excluded from contact-only metrics even if the losing
player still has a checker on the bar.

Use variation across games and seed batches when judging differences: turns
within a game are correlated and must not be treated as independent experimental
replicates. Any later confidence analysis should resample whole games or matched
seed/game groups. This run is exploratory, not final confirmation after tuning.

### Aggressive strength screening

Run `python3 -B sweep_aggressive_strength.py` for strengths 0 (control),
0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, and 1.0. Each plays 100 games
against the frozen original baseline: five shared seed batches of 20 games,
balanced between player assignments (900 games total). Both sides use
20 heuristic-guided rollouts and depth 4 with the original heuristic weights.
Three workers run by default. Results are saved after every game under
`data/aggressive_strength_sweep`; restart the same command with `--resume`
only after the original process stops. Resume requires unchanged settings,
source files, and Python version.

Compare complete win rates and uncertainty, hits when available, hits per
turn, and MC utility sacrificed. Strength is a maximum normalized bonus,
not a bonus per hit. This screening does not change defaults. Confirm any
selected strength against zero and 0.05 using fresh held-out seeds before
adopting it; the largest observed screening win rate is not proof of an optimum.

### Rear-priority strength screening

Run `python3 -B sweep_rear_priority_strength.py` to test the same nine
strengths (zero control plus 0.05 through 1.0) against the frozen baseline.
Five fresh seed batches (1181000 through 1185000 in increments of 1000)
give 100 games per strength, balanced between seats, or 900 total.
The search uses 20 guided rollouts and depth 4. Check win rates together
with advancement choice rank (higher means choosing farther-back checkers)
and MC utility sacrificed. Results checkpoint under
`data/rear_priority_strength_sweep`; use `--resume` after interruption.
Defaults remain unchanged; confirm selected strengths on fresh seeds.

Aggressive strength remains provisionally 0.05 after two 900-game sweeps:
98/200 wins versus the zero control's 100/200, with increased hitting
behavior. These samples do not establish equivalence or optimality.

### Front-priority strength screening

Run `python3 -B sweep_front_priority_strength.py` for the same nine
strengths, including zero, against the frozen baseline. Each strength
plays 100 games across five fresh seed batches (1221000 through 1225000
in increments of 1000), with 50 games in each seat. Both sides retain
20 guided rollouts and depth 4. Results checkpoint under
`data/front_priority_strength_sweep`; restart with `--resume` if interrupted.
Lower advancement choice rank indicates stronger front-checker preference.
Compare this behavior with win rate and estimated MC utility sacrificed;
confirm promising settings on fresh seeds before adoption.

Rear-priority strength remains 0.05 following confirmation: combined
win counts were 101/200 for zero and 100/200 each for 0.05 and 0.10.
The 0.05 setting produces a rear-checker preference while sacrificing
less estimated MC utility than 0.10; optimality is not established.

### Defensive strength screening

Run `python3 -B sweep_defensive_strength.py` for nine strengths:
0, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, and 1.0.
Each plays 100 games against the frozen baseline, using five fresh
seed batches (1281000 through 1285000 in increments of 1000), balanced
between seats. Both sides use 20 guided rollouts and depth 4.
Results checkpoint under `data/defensive_strength_sweep`; use `--resume`
after interruption. Evaluate win rate alongside contact-phase threatened
blots and made points, plus estimated MC utility sacrificed. Confirm
promising settings on fresh seeds before adoption.

Front-priority's provisional selected strength is 0.01. Confirmation
win counts were 50/100 for zero, 49/100 for 0.01, and 44/100 for 0.02;
combined with screening, counts were 99/200, 101/200, and 96/200.
The 0.01 setting produces a consistent front preference with less
estimated MC utility sacrificed than 0.02. Superiority is not established.
These selections document experimental settings; constructor defaults
are not changed automatically by the sweep runners.

### Selected strengths for rule-variation experiments

The constructor and evaluation/benchmark CLI defaults now use rear-priority
0.05, front-priority 0.01, aggressive 0.05, and defensive 0.50.
`--style-strength` remains an explicit override, including zero.
`configs/selected_agents.json` records the selected strengths and experimental
search settings (20 guided rollouts, depth 4, original heuristic).
Use explicit search flags: the ordinary CLI search-budget defaults remain unchanged.
Defensive confirmation against the frozen baseline produced 104/200 wins
for 0.1 and 105/200 for 0.5; 0.5 was selected for its clearer defensive
behavior, not a demonstrated win-rate advantage. Historical sweep manifests
are preserved; their source hashes predate this defaults change.
