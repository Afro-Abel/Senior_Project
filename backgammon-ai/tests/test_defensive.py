import random
import unittest
from unittest.mock import patch

from agents.defensive_agent import DefensiveAgent, made_points
from agents.monte_carlo_agent import MonteCarloAgent, MoveEvaluation
from agents.styled_agent import StyleMoveEvaluation
from backgammon.move import BAR, OFF, Move, Step
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC, contact_features


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


class DefensiveTests(unittest.TestCase):
    def test_rewards_covering_threatened_blots(self):
        state = position({8: 1, 6: 1, 2: -1})
        after = apply_step(state, Step(8, 6), 1)
        self.assertEqual(contact_features(state, 1)[2], 2)
        self.assertEqual(contact_features(after, 1)[2], 0)
        self.assertEqual(DefensiveAgent().raw_style_score(state, after, 1), 3)

    def test_rewards_removing_threat_without_building_point(self):
        state = position({8: 1, 6: 1, 2: -1})
        after = apply_step(state, Step(6, 1), 1)
        self.assertEqual(made_points(after, 1), made_points(state, 1))
        self.assertEqual(DefensiveAgent().raw_style_score(state, after, 1), 1)

    def test_protected_points_include_outside_home_and_anchors(self):
        agent = DefensiveAgent()
        for source, destination in ((10, 8), (22, 20)):
            state = position({source: 1, destination: 1, 1: -1})
            self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(source, destination), 1), 1), 1)

    def test_breaking_point_and_exposing_blots_is_penalized(self):
        state = position({8: 2, 2: -1})
        after = apply_step(state, Step(8, 6), 1)
        self.assertEqual(DefensiveAgent().raw_style_score(state, after, 1), -3)

    def test_mirrored_players_choose_protected_outcome_on_equal_mc(self):
        state = position({8: 1, 7: 1, 1: -1})
        reflected = GameState([-c for c in reversed(state.board)],
                              {p: state.bar[-p] for p in (1, -1)},
                              {p: state.borne_off[-p] for p in (1, -1)}, -1)
        for state in (state, reflected):
            snapshot = state.copy()
            agent = DefensiveAgent(1, random.Random(4))
            with patch("agents.monte_carlo_agent.rollout", return_value=0):
                move = agent.choose_move(state, (1, 2))
            self.assertIn(move, get_legal_moves(state, (1, 2), state.current_player))
            after = apply(state, move)
            point = 6 if state.current_player == 1 else 19
            self.assertEqual(after.board[point - 1], 2 * state.current_player)
            self.assertEqual(state, snapshot)
            self.assertEqual(agent.last_decision_metrics["chosen_normalized_preference"], 1)

    def test_bar_entry_threats_and_own_reentry(self):
        agent = DefensiveAgent()
        state = position({8: 1, 5: 1}, {-1: 1})
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(8, 5), 1), 1), 2)
        state = position({23: -1}, {1: 1})
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(BAR, 24), 1), 1), -1)

    def test_pure_race_and_contact_exit_do_not_reward_blocking(self):
        agent = DefensiveAgent()
        state = position({6: 2, 19: -1})
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(6, 5), 1), 1), 0)
        state = position({7: 2, 6: -1})
        after = apply(state, Move([Step(7, 5), Step(7, 4)]))
        self.assertEqual(made_points(state, 1), 1)
        self.assertEqual(made_points(after, 1), 0)
        self.assertEqual(agent.raw_style_score(state, after, 1), 0)

    def test_equivalent_sequences_dedup_and_equal_preferences(self):
        state = position({8: 1, 7: 1, 1: -1})
        first = apply(state, Move([Step(8, 6), Step(7, 6)]))
        second = apply(state, Move([Step(7, 6), Step(8, 6)]))
        self.assertEqual(first, second)
        agent = DefensiveAgent(3)
        self.assertEqual(agent.raw_style_score(state, first, 1), agent.raw_style_score(state, second, 1))
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as simulate:
            results = agent.evaluate_moves(GameState.initial_state(), (2, 6))
        self.assertEqual(len(results), 14)
        self.assertEqual(simulate.call_count, 42)
        self.assertTrue(all(0 <= r.normalized_style_score <= 1 for r in results))
        self.assertTrue(all(0 <= r.style_adjustment <= agent.style_strength for r in results))

    def test_zero_strength_matches_baseline_rng_and_choice(self):
        state = position({8: 1, 7: 1, 1: -1})
        base = MonteCarloAgent(2, random.Random(8), 1, "heuristic")
        agent = DefensiveAgent(2, random.Random(8), 1, "heuristic", style_strength=0)
        self.assertEqual(base.choose_move(state, (1, 2)), agent.choose_move(state, (1, 2)))
        self.assertEqual(base.rng.getstate(), agent.rng.getstate())
        self.assertEqual(agent.heuristic_config, DEFAULT_HEURISTIC)

    def test_mc_can_outweigh_preference(self):
        state = position({8: 1, 7: 1, 1: -1})
        safe = Move([Step(8, 6), Step(7, 6)])
        exposed = Move([Step(7, 5), Step(8, 7)])
        for utility, expected in ((.02, safe), (.2, exposed)):
            with patch.object(MonteCarloAgent, "evaluate_moves", return_value=[
                MoveEvaluation(safe, 0, 20), MoveEvaluation(exposed, utility, 20)]):
                self.assertEqual(DefensiveAgent(style_strength=.05).choose_move(state, (1, 2)), expected)

    def test_terminal_priority_and_pass(self):
        win, other = Move([Step(1, OFF)]), Move([Step(3, 1)])
        agent = DefensiveAgent(style_strength=100)
        with patch.object(agent, "evaluate_moves", return_value=[
            StyleMoveEvaluation(win, 1, 0, immediate_win=True, mc_score=1),
            StyleMoveEvaluation(other, 100, 20, mc_score=.9)]):
            self.assertEqual(agent.choose_move(position({1: 1, 22: -1}), (1, 2)), win)
        self.assertIsNone(agent.choose_move(position({24: -2, 23: -2}, {1: 1}), (1, 2)))
        self.assertIsNone(agent.last_decision_metrics)

    def test_evaluator_behavior_labels_in_both_seats(self):
        from evaluate_baseline import play_game, summarize
        records = [play_game(i, 1106100, 1, 0, 10000, "random", "heuristic",
                             DEFAULT_HEURISTIC, "defensive", .05) for i in (0, 1)]
        self.assertEqual([r["mc_player"] for r in records], [1, -1])
        self.assertTrue(all(r["agent"] == "defensive" for r in records))
        self.assertEqual(summarize(records)["style_behavior"]["style"], "defensive")


if __name__ == "__main__":
    unittest.main()
