# AI-Based Playtesting of Backgammon Rule Variations Using Monte Carlo Agents

Midterm Progress Report

Abel Adugna | Advisor: Professor James Glenn | October 18, 2026

## Abstract

Automated playtesting can help investigate how changes to game rules affect strategic choices without requiring extensive human testing. This project asks how small Backgammon rule changes affect the relative performance and observable behavior of agents with different heuristic biases. The intended contribution is a reproducible experimental framework that compares rear-priority, front-priority, aggressive, and defensive styles under a shared Monte Carlo decision method. This midterm report presents the simulator and baseline evaluation needed before the central rule-variation experiments.

The implemented system includes a Backgammon engine, legal complete-turn generation, a terminal interface, automated correctness checks, and seeded complete-game evaluation. The baseline uses flat Monte Carlo search with configurable rollout count and depth. Equivalent resulting positions are evaluated once. Simulations stop at a terminal result or a turn limit, where a shared heuristic evaluates race progress, borne-off checkers, bar checkers, threatened blots, and protected structures. Both random and heuristic-guided rollout policies are available.

The current suite contains 47 passing tests. Experiments show that success against random play alone does not demonstrate useful lookahead: a retained random-rollout configuration won only four of 40 games against heuristic-only play. Guided rollouts using 20 simulations per candidate and depth four won 53 of 80 games across two retained seed batches, with an approximate 95 percent Wilson interval of 55.4 to 75.7 percent. Mean decision time was approximately 0.323 seconds. Five-rollout guided configurations won 23 of 80 games, suggesting sensitivity to sampling budget, although seed differences limit causal interpretation.

These exploratory results support a provisional baseline rather than an optimal configuration. The main research question remains unanswered: the four styles and rule variations have not yet been implemented. Remaining work will validate behavioral differences, select hypothesis-driven variations, and conduct controlled tournaments.

## 1. Introduction

Playtesting is essential for understanding how game rules influence player choices and outcomes. Human testing provides valuable feedback, but repeatedly evaluating possible rule changes requires time and participants. Automated agents can complement that process by generating repeatable games and quantitative observations. Previous work has used agents with different styles to study board-game configurations and has used self-play to investigate alternative chess rules [1, 2].

The research question for this project is: How do small Backgammon rule changes affect the relative performance and observable behavior of Monte Carlo agents with different heuristic biases? The aim is to build an experimental environment that can expose changes in strategic incentives, rather than to produce an expert-strength Backgammon player.

Four intended styles are rear-priority, front-priority, aggressive, and defensive. They will respectively favor advancing trailing checkers, advancing leading checkers, hitting opposing blots, and protecting checkers or building safe positions. All four will share the same underlying decision method and configured search budget. Before introducing rule variations, their intended behavioral differences must be measured under standard rules.

At midterm, the main achievements are an operational Backgammon engine, automated rule and state checks, configurable flat Monte Carlo search, a shared cutoff heuristic, and experiments comparing random and heuristic-guided rollout policies. These establish the platform and initial baseline needed for the central research experiments. No conclusions about rule variations or the four planned styles have been obtained yet.

## 2. Background

Backgammon combines visible board positions with stochastic dice outcomes. Each player moves 15 checkers around 24 points toward a home board and then bears them off. A point occupied by two or more opposing checkers is blocked. Landing on a lone opposing checker, or blot, sends it to the bar. Under the implemented standard movement rules, bar checkers must re-enter before other checkers move. Legal complete turns must use the maximum possible number of dice, with the higher die required when only one of two different dice can be played. Doubles allow four uses of the rolled value.

A decision occurs after the dice are known. Flat Monte Carlo search evaluates each legal candidate by sampling future play and averaging its outcomes. Unlike Monte Carlo Tree Search, it does not maintain and progressively expand a tree or use a tree-selection rule. The survey by Browne et al. provides broader context for simulation-based tree-search methods [3]; the present implementation uses the simpler flat approach.

Full-game simulations were implemented initially, but their cost motivated a configurable depth limit. A depth of four means four individual player turns after the candidate move: opponent, original player, opponent, original player. A forced pass consumes one turn. If a simulation has not ended at the limit, a position heuristic supplies its score. This is a utility estimate, not a calibrated winning probability.

## 3. Methodology

### 3.1. Platform and game engine

The implementation uses Python and its standard library. The state contains a signed list of 24 point counts, separate bar and borne-off counts for each player, and the current player. Positive counts represent Player 1, who moves toward point 1; negative counts represent Player 2, who moves toward point 24. Independent state copies allow hypothetical moves without changing the actual game.

The code separates state representation, move structures, rules, game flow, agents, and evaluation. The rules module generates complete legal movement sequences. The game layer applies turns, switches players, rolls dice, and checks winners. The opening contest rolls one die per player, repeats ties, and lets the higher roller start using both dice. A terminal interface supports manual inspection and human play. Doubling-cube and match-scoring behavior are outside the current prototype.

The decision pipeline is:

Actual state and dice -> legal move sequences -> distinct resulting states -> bounded rollouts -> mean utility -> selected legal move.

Different sequences can produce the same complete state. The agent deduplicates these outcomes using the board, bar, borne-off counts, and current player, retaining one legal representative. For the tested opening roll of 2 and 6 for Player 2, 28 legal sequences reduce to 14 distinct outcomes. Thus, 20 rollouts require 280 simulations for that decision, rather than 560. The rules engine retains the original legal sequences for validation and other callers.

### 3.2. Flat Monte Carlo decision method

For a state $s$, observed dice $r$, and candidate action $a$, the agent estimates:

$$
\widehat{Q}(s,a)=\frac{1}{N}\sum_{i=1}^{N}V_p(s_{i,d}),
$$

where $N$ is the rollout count, $p$ is the player making the original decision, and $s_{i,d}$ is the state reached after at most $d$ subsequent player turns. The agent chooses an action with the highest average score, breaking ties randomly. Immediate winning candidates receive an exact score without simulation. A forced pass returns no action.

Each decision follows this algorithm:

1. Generate legal complete turns for the actual dice.
2. Apply each turn to a copied state and retain distinct outcomes.
3. For each nonterminal outcome, begin with the opponent to move.
4. Run $N$ simulations, sampling dice and selecting moves under the chosen rollout policy for at most $d$ turns.
5. Score each ending state from the original player's perspective and average the scores.
6. Return a legal representative of a highest-scoring outcome.

Two rollout policies are implemented. The random policy samples legal move sequences uniformly. The guided policy evaluates distinct immediate outcomes from the simulated acting player's perspective and chooses a highest-scoring move, with random tie-breaking. Guided rollouts do not invoke another Monte Carlo search. Both policies use random dice and the same final cutoff heuristic.

### 3.3. Shared position heuristic

The heuristic combines race progress with contact-related structure. Let $P$ denote pip count, $O$ borne-off checkers, $B$ bar checkers, $H$ made home points, $L$ longest consecutive run of made points, and $T$ directly threatened blots. Subscripts $p$ and $o$ refer to the original player and opponent. The raw score is:

$$
R=(P_o-P_p)+4(O_p-O_o)
$$

and, while contact remains possible, it additionally includes:

$$
8(B_o-B_p)+2(H_p-H_o)+1.5(L_p^2-L_o^2)+3(T_o-T_p).
$$

Pip count measures remaining checker travel and assigns 25 pips to each bar checker. Made points contain at least two own checkers. Threatened blots use a direct single-die reachability proxy, including bar entry, rather than exact hitting probabilities; combined-dice hits are omitted. Once the players have passed each other and neither has bar checkers, contact terms are disabled.

For nonterminal states, the returned utility is $V_p(s)=0.99\tanh(R/50)$. A completed win returns $+1$ and a completed loss returns $-1$. The weights are provisional design choices, not learned or empirically optimized coefficients. In particular, the borne-off and bar terms intentionally supplement effects already represented in pip count. Their usefulness still requires sensitivity testing.

### 3.4. Evaluation and reproducibility

The evaluator alternates the main agent between Player 1 and Player 2, while the opening contest independently decides who starts. Each game has separate seeded streams for actual dice, the main agent, and its opponent. Identical code, settings, and seeds reproduce game outcomes; decision timings vary. Repeating an identical seed batch is a replay, not new evidence.

Reports save settings, source hashes, seeds, game results, and per-decision timing. Checks detect illegal actions, incorrect forced passes or dice usage, mutation of the actual state during search, checker-count violations, and invalid terminal outcomes. Exceeding the game turn limit raises an error rather than recording a draw. The random opponent samples sequences, while heuristic candidates are deduplicated; these are distinct policies and should not be interpreted as equivalent sampling schemes.

## 4. Baseline Validation and the Effects of Rollout Design

### 4.1. Rule and State Checks Support Correct Complete-Game Execution

The automated suite contains 47 passing tests covering movement rules, opening rolls, outcome deduplication, heuristic symmetry, rollout depth, turn alternation, and scoring perspective. Rule fixtures check both players through mirrored positions. An earlier 100-game random-versus-random run completed without state or termination failures. Together, these results support using the engine for pilot experiments. They do not establish exhaustive correctness or agreement with an independent expert engine.

Table 1 summarizes the retained complete-game experiments. The main agent alternates equally between the two player identities within each batch; the opening roll independently determines who starts. Report settings, rather than filenames, identify the configurations.

Table 1. Complete-game performance of the evaluated agent against random and heuristic-only opponents. Wins and win rates refer to the evaluated agent. A rollout is one simulated continuation per distinct candidate; depth counts subsequent individual player turns, excluding the candidate, and simulations stop early at a terminal state. Guided moves maximize the acting player's immediate heuristic score; heuristic-only play evaluates candidates without simulating future turns. Each 40-game batch assigns the evaluated agent to each player identity 20 times. Rows pool different seed batches and vary in policy, rollout count, or depth, so the table is an exploratory summary rather than an isolated comparison of any single factor.

| Opponent | Evaluated policy | Rollouts per candidate | Depth in turns | Wins / games | Win rate |
| --- | --- | --- | --- | --- | --- |
| Random | Random rollouts | 1 | 1 | 159 / 160 | 99.4% |
| Random | Heuristic-only | Not applicable | 0 | 40 / 40 | 100.0% |
| Heuristic-only | Random rollouts | 20 | 20 | 4 / 40 | 10.0% |
| Heuristic-only | Guided rollouts | 20 | 4 | 53 / 80 | 66.3% |
| Heuristic-only | Guided rollouts | 5 | 4 | 23 / 80 | 28.8% |

The first row pools base seeds 20261017, 5000, 6000, and 7000. The heuristic-only and random-depth-20 rows each use seed 20261017. The guided 20-rollout row pools seeds 20261017 and 5000; the guided five-rollout row pools seeds 10000 and 11000.

### 4.2. Success Against Random Play Does Not Establish the Value of Lookahead

The shallow random-rollout agent won 159 of 160 games against random play. However, heuristic-only play also won all 40 games in its random-opponent batch. These outcomes show an advantage over random choices but leave little room to distinguish the value of search. Against heuristic-only play, the retained random-rollout configuration with 20 rollouts and depth 20 won only four of 40 games. Thus, success against a weak opponent was insufficient validation of the search method.

A plausible explanation is that random continuations evaluate positions under behavior unlike the actual opponent's. Actions that benefit from random mistakes need not perform well against deliberate heuristic choices. This interpretation concerns the rollout model, not a demonstrated failure of Monte Carlo search in general. The experiments do not isolate the mechanism, and the retained configurations differ in depth and opponent.

### 4.3. Guided Rollouts Show a Promising Advantage Over Immediate Evaluation

At depth four with 20 rollouts per candidate, guided search won 29/40 games in one retained batch and 24/40 in another. The pooled result is 53/80 wins, or 66.3%, with an approximate 95% Wilson interval of 55.4%-75.7%. This supports a provisional advantage over the shared immediate heuristic in this matchup, while the variation between batches shows why one batch is insufficient.

Guided rollouts choose sensible immediate moves for each simulated player while retaining stochastic dice and random tie-breaking. They therefore more closely match the heuristic-only opponent than random continuations do. This offers a plausible explanation for the improvement, although the random and guided rows in Table 1 also differ in depth. A matched policy comparison is needed before attributing the difference solely to rollout guidance.

This finding motivates validating agent behavior before studying rule variations. It is consistent with the methodological emphasis on agent-based playtesting in de Mesentier Silva et al. [1], but their Ticket to Ride setting differs from this project. The present win rates are not directly comparable to that study or to AlphaZero's chess-variant results [2]. No modified Backgammon rules have been tested yet.

### 4.4. Five-Rollout Results Suggest Sensitivity to Sampling Budget

With guided rollouts and the same depth of four, reducing the budget to five rollouts produced 12/40 and 11/40 wins, totaling 23/80 or 28.8%. One possible explanation is noisy action ranking: a small sample can make an inferior action appear favorable, especially when the agent selects the highest estimate among many candidates.

The observed difference from the 20-rollout results is large, but the pooled groups use different seed batches. These experiments therefore suggest budget sensitivity rather than establish a causal effect or an optimal rollout count. The 20-rollout configuration remains the provisional choice; a matched seed comparison and held-out evaluation should precede stronger claims.

### 4.5. Typical Decision Times Are Practical but Occasional Decisions Are Slow

The retained 20-rollout, depth-four guided batches recorded 3,215 main-agent decisions, including forced passes. Their pooled mean decision time was approximately 0.323 seconds, calculated over decisions rather than averaging the two batch means equally. The individual batch means were 0.347 and 0.299 seconds. Their within-batch 95th-percentile times were 1.273 and 1.148 seconds, while maximum times reached 16.002 and 16.367 seconds.

Typical decisions are fast enough for continued development, but the long tail matters when planning large tournaments. Cost depends on legal candidate counts and the simulated positions encountered. Guided rollouts evaluate candidate positions at each simulated turn, so equal rollout counts across policies do not represent equal computational budgets. These local measurements exclude the evaluator's legality and state-checking overhead; total experiment time is greater.

### 4.6. Preserved Evidence Supports a Provisional Baseline, Not Final Research Conclusions

The current operational baseline is guided rollouts, depth four, and 20 rollouts per distinct candidate. These settings are neither an established optimum nor evidence of expert playing strength. The heuristic weights remain hand-selected, and the reported intervals are exploratory because the experiments informed configuration selection.

An additional guided 20-rollout seed-10000 batch was reported as 26/40 wins in terminal output, but its JSON was overwritten by a five-rollout run. It is excluded from Table 1 and the retained-data aggregate. Two filenames containing "r20" currently record five rollouts; their settings determine how they are classified. Unique experiment identifiers and an experiment ledger are needed before formal data collection.

The midterm results establish an operational simulator and a promising search baseline. They do not yet establish distinct strategic styles or the effects of rule changes. Those questions require behavioral validation of the four planned agents followed by controlled standard-rule and modified-rule tournaments.

## 5. Related Work

De Mesentier Silva et al. use four agents with different playing styles to analyze Ticket to Ride maps and rule configurations [1]. Their work motivates studying multiple strategic preferences rather than relying on one general agent. This project applies that perspective to Backgammon and will compare both relative performance and measured behavior.

Tomasev et al. use AlphaZero to investigate small changes to chess rules and analyze their effects on gameplay [2]. Their study supports using automated play to explore rule consequences. This project differs in its stochastic game, smaller computational scope, and planned comparison of deliberately biased styles.

Browne et al. survey Monte Carlo Tree Search and its variants [3]. Their work provides context for simulation-based decision methods and the distinction between sampling and tree search. The present baseline uses equal-budget flat Monte Carlo evaluation, without UCT, adaptive tree expansion, or learned value networks. These sources motivate the methodology; they do not validate this project's hand-selected heuristic weights.

## 6. Conclusions

The project asks how small Backgammon rule changes affect agents with different strategic priorities. At midterm, the simulator and evaluation infrastructure are operational, and a depth-limited flat Monte Carlo agent has been implemented and tested. Experiments show why random-opponent success alone is insufficient validation: random-rollout lookahead struggled against heuristic-only play, while guided rollouts achieved 53 wins in 80 retained games at the provisional baseline settings. These findings support further development but do not yet answer the central rule-variation question. The remainder of the project will implement and validate the four styles, finalize hypothesis-driven rule variations, and conduct controlled tournaments.

## 7. Future Work

The environment milestone is substantially complete, and a working baseline has undergone initial validation. The project is transitioning from the proposed weeks 5-6 baseline milestone toward weeks 7-8, which focus on strategic biases. Final rule variations and style definitions were deferred rather than completed during the originally proposed weeks 1-2.

The next task is to document a fixed baseline and design the four strategy-specific biases. A proposed approach is to add small configurable adjustments to candidate scores while keeping the cutoff heuristic and rollout policy shared. Behavioral metrics must establish that rear-priority, front-priority, aggressive, and defensive agents actually differ as intended. Equal search settings will be maintained, with measured runtime reported because candidate counts can differ.

After those pilots, rule variations will be selected around explicit strategic incentives and prewritten hypotheses. Each variation should change a clearly identified mechanism, and experiments should measure how often it affects play. The proposed scope is two core variations initially, with additional variations only if time permits. The exact choices remain undecided.

Later work will run balanced tournaments under standard and modified rules, collecting win rates, game lengths, hits, pip-count progression, exposed blots, and rule-activation frequency. Results will be compared against standard-rule baselines and accompanied by uncertainty estimates. Pilot seeds will be separated from final evaluation seeds. Final deliverables remain the documented simulator and agents, gameplay datasets, analysis, report, and presentation.

Key remaining challenges are heuristic-weight justification, rollout cost, preservation of experiment records, and separating strategic style from playing strength. Optional comparison with GNU Backgammon remains future work. Conclusions will be limited to the implemented agents and tested variations rather than universal claims about human Backgammon strategy.

## References

[1] F. de Mesentier Silva, S. Lee, J. Togelius, and A. Nealen. "AI-based Playtesting of Contemporary Board Games." Proceedings of the 12th International Conference on the Foundations of Digital Games, 2017. DOI: https://doi.org/10.1145/3102071.3102105. Author-hosted paper: https://www.nealen.net/papers/3102071.3102105.pdf.

[2] N. Tomasev, U. Paquet, D. Hassabis, and V. Kramnik. "Assessing Game Balance with AlphaZero: Exploring Alternative Rule Sets in Chess." arXiv:2009.04374, 2020. https://arxiv.org/abs/2009.04374.

[3] C. B. Browne et al. "A Survey of Monte Carlo Tree Search Methods." IEEE Transactions on Computational Intelligence and AI in Games, 4(1):1-43, 2012. DOI: https://doi.org/10.1109/TCIAIG.2012.2186810.
