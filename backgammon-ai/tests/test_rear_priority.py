import random
import unittest
from unittest.mock import patch

from agents.monte_carlo_agent import MonteCarloAgent, MoveEvaluation
from agents.rear_priority_agent import RearPriorityAgent
from agents.styled_agent import StyleMoveEvaluation
from backgammon.move import BAR, OFF, Move, Step
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC


def position(points, bar=0):
    state = GameState()
    for p, count in points.items():
        state.board[p - 1] = count
    state.bar[1] = bar
    for p in (1, -1):
        state.borne_off[p] = 15 - state.checker_count(p)
    return state


def after_move(state, move):
    after = state.copy()
    for step in move.steps:
        after = apply_step(after, step, state.current_player)
    return after


class RearPriorityTests(unittest.TestCase):
    def test_equal_advance_rewards_trailing_checker_more(self):
        state = position({20: 1, 8: 1, 19: -1})
        agent = RearPriorityAgent()
        rear = apply_step(state, Step(20, 18), 1)
        front = apply_step(state, Step(8, 6), 1)
        self.assertEqual(agent.raw_style_score(state, rear, 1), 76)
        self.assertEqual(agent.raw_style_score(state, front, 1), 28)

    def test_equal_mc_utilities_choose_rear_for_both_players(self):
        state = position({20: 1, 8: 1, 19: -1})
        reflected = GameState([-c for c in reversed(state.board)],
                              {p: state.bar[-p] for p in (1, -1)},
                              {p: state.borne_off[-p] for p in (1, -1)}, -1)
        for current in (state, reflected):
            snapshot = current.copy()
            agent = RearPriorityAgent(1, random.Random(5))
            with patch("agents.monte_carlo_agent.rollout", return_value=0):
                move = agent.choose_move(current, (1, 2))
            self.assertIn(move, get_legal_moves(current, (1, 2), current.current_player))
            result = after_move(current, move)
            destination = 17 if current.current_player == 1 else 8
            self.assertEqual(result.board[destination - 1], current.current_player)
            self.assertEqual(current, snapshot)
            self.assertEqual(agent.last_decision_metrics["chosen_normalized_preference"], 1)

    def test_equivalent_step_orders_have_identical_preference(self):
        state = position({20: 1, 8: 1, 24: -1})
        first = after_move(state, Move([Step(20, 19), Step(19, 17)]))
        second = after_move(state, Move([Step(20, 18), Step(18, 17)]))
        self.assertEqual(first, second)
        agent = RearPriorityAgent()
        self.assertEqual(agent.raw_style_score(state, first, 1),
                         agent.raw_style_score(state, second, 1))

    def test_bar_entry_and_bearing_off_distances(self):
        agent = RearPriorityAgent()
        state = position({19: -1}, bar=1)
        entered = apply_step(state, Step(BAR, 24), 1)
        self.assertEqual(agent.raw_style_score(state, entered, 1), 25 ** 2 - 24 ** 2)
        state = position({3: 1, 22: -1})
        off = apply_step(state, Step(3, OFF), 1)
        self.assertEqual(agent.raw_style_score(state, off, 1), 9)

    def test_zero_bias_preserves_base_scores_rng_and_move(self):
        state = position({6: 2, 19: -2})
        base = MonteCarloAgent(2, random.Random(17), 2, "heuristic")
        rear = RearPriorityAgent(2, random.Random(17), 2, "heuristic", style_strength=0)
        self.assertEqual(base.choose_move(state, (1, 2)), rear.choose_move(state, (1, 2)))
        self.assertEqual(base.rng.getstate(), rear.rng.getstate())

    def test_bonus_can_break_close_scores_but_not_large_mc_advantage(self):
        state = position({20: 1, 8: 1, 19: -1})
        rear = Move([Step(20, 19), Step(19, 17)])
        front = Move([Step(8, 7), Step(7, 5)])
        agent = RearPriorityAgent(style_strength=.05)
        for front_score, expected in ((.02, rear), (.2, front)):
            with patch.object(MonteCarloAgent, "evaluate_moves", return_value=[
                MoveEvaluation(rear, 0, 20), MoveEvaluation(front, front_score, 20)]):
                self.assertEqual(agent.choose_move(state, (1, 2)), expected)
        self.assertEqual(agent.heuristic_config, DEFAULT_HEURISTIC)

    def test_proven_win_cannot_be_displaced_by_bonus(self):
        win = Move([Step(1, OFF)])
        other = Move([Step(3, 1)])
        agent = RearPriorityAgent()
        with patch.object(agent, "evaluate_moves", return_value=[
            StyleMoveEvaluation(win, 1, 0, immediate_win=True, mc_score=1),
            StyleMoveEvaluation(other, 10, 20, mc_score=.9)]):
            self.assertEqual(agent.choose_move(position({1: 1, 22: -1}), (1, 2)), win)

    def test_normalization_equal_outcomes_and_deduplication(self):
        state = position({20: 1, 24: -1})
        with patch("agents.monte_carlo_agent.rollout", return_value=0):
            results = RearPriorityAgent(1).evaluate_moves(state, (1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].normalized_style_score, 0)
        state = GameState.initial_state()
        state.current_player = -1
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as simulate:
            results = RearPriorityAgent(3).evaluate_moves(state, (2, 6))
        self.assertEqual(len(results), 14)
        self.assertEqual(simulate.call_count, 42)
        self.assertTrue(all(0 <= r.normalized_style_score <= 1 for r in results))
        self.assertTrue(all(0 <= r.style_adjustment <= .05 for r in results))

    def test_same_final_own_checker_but_different_hits_remain_distinct(self):
        state = position({20: 1, 19: -1})
        with patch("agents.monte_carlo_agent.rollout", return_value=0):
            results = RearPriorityAgent(1).evaluate_moves(state, (1, 2))
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].raw_style_score, results[1].raw_style_score)
        after = [after_move(state, r.move) for r in results]
        self.assertEqual({s.bar[-1] for s in after}, {0, 1})

    def test_pass_clears_last_metrics(self):
        agent = RearPriorityAgent(1, depth=0)
        agent.choose_move(position({20: 1, 19: -1}), (1, 2))
        self.assertIsNotNone(agent.last_decision_metrics)
        self.assertIsNone(agent.choose_move(position({24: -2, 23: -2}, bar=1), (1, 2)))
        self.assertIsNone(agent.last_decision_metrics)

    def test_invalid_strength(self):
        for strength in (-.1, float("nan"), float("inf"), True, "0.05"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                RearPriorityAgent(style_strength=strength)

    def test_evaluator_records_style_behavior_in_both_seats(self):
        from evaluate_baseline import play_game, summarize
        records = [play_game(index, 1100100, 1, 0, 10000, "random", "heuristic",
                             DEFAULT_HEURISTIC, "rear-priority", .05)
                   for index in (0, 1)]
        self.assertEqual([r["mc_player"] for r in records], [1, -1])
        for record in records:
            self.assertEqual(record["agent"], "rear-priority")
            self.assertTrue(record["style_decisions"])
            self.assertTrue(all(0 <= d["chosen_normalized_preference"] <= 1
                                for d in record["style_decisions"]))
        summary = summarize(records, "random")
        behavior = summary["style_behavior"]
        self.assertEqual(behavior["style"], "rear-priority")
        self.assertEqual(behavior["evaluated_decisions"],
                         sum(len(r["style_decisions"]) for r in records))


if __name__ == "__main__":
    unittest.main()
