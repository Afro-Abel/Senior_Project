import random
import unittest
from unittest.mock import patch

from agents.aggressive_agent import AggressiveAgent
from agents.monte_carlo_agent import MonteCarloAgent, MoveEvaluation
from agents.styled_agent import StyleMoveEvaluation
from backgammon.move import BAR, OFF, Move, Step
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC


def position(points, bars=None):
    state = GameState()
    for point, count in points.items():
        state.board[point - 1] = count
    if bars:
        state.bar.update(bars)
    for p in (1, -1):
        state.borne_off[p] = 15 - state.checker_count(p)
    return state


def apply(state, move):
    for step in move.steps:
        state = apply_step(state, step, state.current_player)
    return state


class AggressiveTests(unittest.TestCase):
    def test_counts_new_hits_not_existing_bar_checkers(self):
        state = position({8: 1, 6: -1, 5: -1}, {-1: 3})
        double = apply(state, Move([Step(8, 6), Step(6, 5)]))
        single = apply(state, Move([Step(8, 7), Step(7, 5)]))
        agent = AggressiveAgent()
        self.assertEqual(agent.raw_style_score(state, double, 1), 2)
        self.assertEqual(agent.raw_style_score(state, single, 1), 1)
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(8, 7), 1), 1), 0)

    def test_prefers_more_hits_for_both_players_on_equal_mc(self):
        state = position({8: 1, 6: -1, 5: -1})
        reflected = GameState([-c for c in reversed(state.board)],
                              {p: state.bar[-p] for p in (1, -1)},
                              {p: state.borne_off[-p] for p in (1, -1)}, -1)
        for state in (state, reflected):
            before = state.copy()
            agent = AggressiveAgent(1, random.Random(5))
            with patch("agents.monte_carlo_agent.rollout", return_value=0):
                move = agent.choose_move(state, (1, 2))
            self.assertIn(move, get_legal_moves(state, (1, 2), state.current_player))
            self.assertEqual(apply(state, move).bar[-state.current_player], 2)
            self.assertEqual(state, before)
            self.assertEqual(agent.last_decision_metrics["chosen_normalized_preference"], 1)

    def test_bar_entry_hit_and_nonhitting_bearing_off(self):
        state = position({24: -1}, {1: 1})
        agent = AggressiveAgent()
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(BAR, 24), 1), 1), 1)
        state = position({3: 1, 22: -1})
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(3, OFF), 1), 1), 0)

    def test_paths_with_different_hits_stay_distinct(self):
        state = position({8: 1, 6: -1, 5: -1})
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as simulate:
            results = AggressiveAgent(3).evaluate_moves(state, (1, 2))
        self.assertEqual(len(results), 2)
        self.assertEqual(simulate.call_count, 6)
        self.assertEqual({r.raw_style_score for r in results}, {1, 2})
        self.assertEqual({r.normalized_style_score for r in results}, {0, 1})
        self.assertEqual({r.style_adjustment for r in results}, {0, .05})

    def test_no_hitting_opportunity_has_no_bonus(self):
        with patch("agents.monte_carlo_agent.rollout", return_value=0):
            results = AggressiveAgent(1).evaluate_moves(GameState.initial_state(), (1, 2))
        self.assertTrue(results)
        self.assertTrue(all(r.raw_style_score == 0 and r.style_adjustment == 0 for r in results))

    def test_zero_bias_preserves_original_mc_choice_and_rng(self):
        state = position({8: 1, 6: -1, 5: -1})
        base = MonteCarloAgent(2, random.Random(6), 1, "heuristic")
        agent = AggressiveAgent(2, random.Random(6), 1, "heuristic", style_strength=0)
        self.assertEqual(base.choose_move(state, (1, 2)), agent.choose_move(state, (1, 2)))
        self.assertEqual(base.rng.getstate(), agent.rng.getstate())
        self.assertEqual(agent.heuristic_config, DEFAULT_HEURISTIC)

    def test_close_scores_can_change_but_strong_mc_advantage_wins(self):
        state = position({8: 1, 6: -1, 5: -1})
        double = Move([Step(8, 6), Step(6, 5)])
        single = Move([Step(8, 7), Step(7, 5)])
        for score, expected in ((.02, double), (.2, single)):
            with patch.object(MonteCarloAgent, "evaluate_moves", return_value=[
                MoveEvaluation(double, 0, 20), MoveEvaluation(single, score, 20)]):
                self.assertEqual(AggressiveAgent().choose_move(state, (1, 2)), expected)

    def test_immediate_win_beats_any_style_bonus(self):
        win, other = Move([Step(1, OFF)]), Move([Step(3, 1)])
        agent = AggressiveAgent(style_strength=100)
        with patch.object(agent, "evaluate_moves", return_value=[
            StyleMoveEvaluation(win, 1, 0, immediate_win=True, mc_score=1),
            StyleMoveEvaluation(other, 100, 20, mc_score=.9)]):
            self.assertEqual(agent.choose_move(position({1: 1, 22: -1}), (1, 2)), win)

    def test_pass_and_report_behavior_for_both_seats(self):
        from evaluate_baseline import play_game, summarize
        agent = AggressiveAgent(1, depth=0)
        self.assertIsNone(agent.choose_move(position({24: -2, 23: -2}, {1: 1}), (1, 2)))
        self.assertIsNone(agent.last_decision_metrics)
        records = [play_game(i, 1104100, 1, 0, 10000, "random", "heuristic",
                             DEFAULT_HEURISTIC, "aggressive", .05) for i in (0, 1)]
        self.assertEqual([r["mc_player"] for r in records], [1, -1])
        self.assertTrue(all(r["agent"] == "aggressive" for r in records))
        self.assertEqual(summarize(records)["style_behavior"]["style"], "aggressive")


if __name__ == "__main__":
    unittest.main()
