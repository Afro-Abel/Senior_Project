"""Candidate-only style adjustments over the unchanged shared MC evaluation."""
from dataclasses import dataclass
from math import isfinite
import random

from agents.style_defaults import SELECTED_STYLE_STRENGTHS
from agents.monte_carlo_agent import MonteCarloAgent, MoveEvaluation
from backgammon.rules import apply_step
from monte_carlo.heuristic import DEFAULT_HEURISTIC


@dataclass(frozen=True)
class StyleMoveEvaluation(MoveEvaluation):
    mc_score: float = 0.0
    raw_style_score: float = 0.0
    normalized_style_score: float = 0.0
    style_adjustment: float = 0.0


class StyleBiasedAgent(MonteCarloAgent):
    style_name = "unspecified"

    def __init__(self, rollouts_per_move=20, rng: random.Random | None = None,
                 depth=4, rollout_policy="heuristic", heuristic_config=DEFAULT_HEURISTIC,
                 style_strength=None):
        if style_strength is None:
            style_strength = SELECTED_STYLE_STRENGTHS.get(self.style_name, 0.05)
        if isinstance(style_strength, bool) or not isinstance(style_strength, (int, float)) \
                or not isfinite(style_strength) or style_strength < 0:
            raise ValueError("style_strength must be a finite nonnegative number")
        super().__init__(rollouts_per_move, rng, depth, rollout_policy, heuristic_config)
        self.style_strength = style_strength
        self.last_decision_metrics = None

    def raw_style_score(self, before, after, player):
        raise NotImplementedError

    def evaluate_moves(self, state, dice):
        # Reuse identical candidates, budgets, RNG draws, and cutoff scores.
        base = super().evaluate_moves(state, dice)
        raw = []
        for candidate in base:
            after = state.copy()
            for step in candidate.move.steps:
                after = apply_step(after, step, state.current_player)
            raw.append(self.raw_style_score(state, after, state.current_player))
        if not raw:
            return []
        low, high = min(raw), max(raw)
        results = []
        for candidate, preference in zip(base, raw):
            normalized = (preference - low) / (high - low) if high > low else 0.0
            bonus = self.style_strength * normalized
            results.append(StyleMoveEvaluation(
                move=candidate.move, score=candidate.score + bonus,
                rollouts=candidate.rollouts, immediate_win=candidate.immediate_win,
                mc_score=candidate.score, raw_style_score=preference,
                normalized_style_score=normalized, style_adjustment=bonus))
        return results

    def choose_move(self, state, dice):
        self.last_decision_metrics = None
        candidates = self.evaluate_moves(state, dice)
        if not candidates:
            return None
        # A style bonus may exceed 1 in experiments; never displace a proven win.
        eligible = [c for c in candidates if c.immediate_win] or candidates
        best = max(c.score for c in eligible)
        chosen = self.rng.choice([c for c in eligible if c.score == best])
        low = min(c.raw_style_score for c in candidates)
        high = max(c.raw_style_score for c in candidates)
        sacrifice = max(c.mc_score for c in candidates) - chosen.mc_score
        self.last_decision_metrics = {
            "style": self.style_name, "candidate_count": len(candidates),
            "preference_range": high - low,
            "chosen_normalized_preference": chosen.normalized_style_score,
            "style_adjustment": chosen.style_adjustment,
            "mc_score": chosen.mc_score, "adjusted_score": chosen.score,
            "mc_utility_sacrificed": sacrifice,
            "style_overrode_mc": sacrifice > 1e-12,
            "immediate_win": chosen.immediate_win}
        return chosen.move
