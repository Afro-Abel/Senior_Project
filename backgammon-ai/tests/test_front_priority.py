import random
import unittest
from unittest.mock import patch

from agents.front_priority_agent import FrontPriorityAgent
from agents.monte_carlo_agent import MonteCarloAgent, MoveEvaluation
from agents.rear_priority_agent import RearPriorityAgent
from backgammon.move import BAR, OFF, Move, Step
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC, pip_count


def position(points, bar=0):
    state = GameState()
    for point, count in points.items():
        state.board[point - 1] = count
    state.bar[1] = bar
    for player in (1, -1):
        state.borne_off[player] = 15 - state.checker_count(player)
    return state


def apply(state, move):
    for step in move.steps:
        state = apply_step(state, step, state.current_player)
    return state


class FrontPriorityTests(unittest.TestCase):
    def test_equal_progress_favors_leading_checker(self):
        state = position({20: 1, 8: 1, 24: -1})
        rear = apply_step(state, Step(20, 18), 1)
        front = apply_step(state, Step(8, 6), 1)
        agent = FrontPriorityAgent()
        self.assertEqual(agent.raw_style_score(state, rear, 1), 28)
        self.assertEqual(agent.raw_style_score(state, front, 1), 76)

    def test_opposite_rear_front_choices_with_equal_mc_for_both_players(self):
        state = position({20: 1, 8: 1, 24: -1})
        reflected = GameState([-c for c in reversed(state.board)],
                              {p: state.bar[-p] for p in (1, -1)},
                              {p: state.borne_off[-p] for p in (1, -1)}, -1)
        for state in (state, reflected):
            snapshot = state.copy()
            with patch("agents.monte_carlo_agent.rollout", return_value=0):
                front = FrontPriorityAgent(1, random.Random(7)).choose_move(state, (1, 2))
                rear = RearPriorityAgent(1, random.Random(7)).choose_move(state, (1, 2))
            front_after = apply(state, front)
            rear_after = apply(state, rear)
            p = state.current_player
            front_destination = 5 if p == 1 else 20
            rear_destination = 17 if p == 1 else 8
            self.assertEqual(front_after.board[front_destination - 1], p)
            self.assertEqual(rear_after.board[rear_destination - 1], p)
            self.assertEqual(state, snapshot)

    def test_equivalent_sequences_and_complementary_scores(self):
        state = position({20: 1, 8: 1, 24: -1})
        first = apply(state, Move([Step(8, 7), Step(7, 5)]))
        second = apply(state, Move([Step(8, 6), Step(6, 5)]))
        agent = FrontPriorityAgent()
        self.assertEqual(first, second)
        self.assertEqual(agent.raw_style_score(state, first, 1), agent.raw_style_score(state, second, 1))
        for move in get_legal_moves(state, (1, 2), 1):
            after = apply(state, move)
            front = agent.raw_style_score(state, after, 1)
            rear = RearPriorityAgent().raw_style_score(state, after, 1)
            self.assertEqual(front + rear, 52 * (pip_count(state, 1) - pip_count(after, 1)))

    def test_bar_entry_and_bearing_off(self):
        agent = FrontPriorityAgent()
        state = position({19: -1}, bar=1)
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(BAR, 24), 1), 1), 3)
        state = position({3: 1, 22: -1})
        self.assertEqual(agent.raw_style_score(state, apply_step(state, Step(3, OFF), 1), 1), 147)

    def test_zero_strength_is_baseline_with_same_rng(self):
        state = position({6: 2, 19: -2})
        base = MonteCarloAgent(2, random.Random(19), 2, "heuristic")
        front = FrontPriorityAgent(2, random.Random(19), 2, "heuristic", style_strength=0)
        self.assertEqual(base.choose_move(state, (1, 2)), front.choose_move(state, (1, 2)))
        self.assertEqual(base.rng.getstate(), front.rng.getstate())

    def test_mc_can_override_preference_and_terminal_wins_take_priority(self):
        state = position({20: 1, 8: 1, 24: -1})
        front = Move([Step(8, 7), Step(7, 5)])
        rear = Move([Step(20, 19), Step(19, 17)])
        agent = FrontPriorityAgent(style_strength=.05)
        for rear_score, expected in ((.02, front), (.2, rear)):
            with patch.object(MonteCarloAgent, "evaluate_moves", return_value=[
                MoveEvaluation(front, 0, 20), MoveEvaluation(rear, rear_score, 20)]):
                self.assertEqual(agent.choose_move(state, (1, 2)), expected)
        state = position({1: 1, 3: 1, 22: -1})
        with patch("agents.monte_carlo_agent.rollout", return_value=1):
            chosen = FrontPriorityAgent(style_strength=100).choose_move(state, (3, 1))
        self.assertEqual(apply(state, chosen).borne_off[1], 15)

    def test_normalization_deduplication_and_forced_pass(self):
        state = position({20: 1, 24: -1})
        agent = FrontPriorityAgent(1)
        with patch("agents.monte_carlo_agent.rollout", return_value=0):
            results = agent.evaluate_moves(state, (1, 2))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].style_adjustment, 0)
        initial = GameState.initial_state()
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as simulate:
            results = FrontPriorityAgent(3).evaluate_moves(initial, (2, 6))
        self.assertEqual(len(results), 14)
        self.assertEqual(simulate.call_count, 42)
        self.assertTrue(all(0 <= r.style_adjustment <= .05 for r in results))
        self.assertIsNone(agent.choose_move(position({24: -2, 23: -2}, bar=1), (1, 2)))
        self.assertIsNone(agent.last_decision_metrics)

    def test_evaluator_labels_and_behavior_for_both_seats(self):
        from evaluate_baseline import play_game, summarize
        records = [play_game(i, 1102100, 1, 0, 10000, "random", "heuristic",
                             DEFAULT_HEURISTIC, "front-priority", .05) for i in (0, 1)]
        self.assertEqual([r["mc_player"] for r in records], [1, -1])
        self.assertTrue(all(r["agent"] == "front-priority" for r in records))
        self.assertEqual(summarize(records)["style_behavior"]["style"], "front-priority")


if __name__ == "__main__":
    unittest.main()
