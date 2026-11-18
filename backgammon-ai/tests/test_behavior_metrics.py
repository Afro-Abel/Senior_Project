import unittest

from backgammon.move import Move, Step
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from behavior_metrics import contact, observe_turn, outcome_features, summarize_turns


def state(points):
    s = GameState()
    for point, count in points.items():
        s.board[point - 1] = count
    for p in (1, -1):
        s.borne_off[p] = 15 - s.checker_count(p)
    return s


def apply(s, move):
    for step in move.steps:
        s = apply_step(s, step, s.current_player)
    return s


class BehaviorTests(unittest.TestCase):
    def test_separates_actor_hits_and_available_opportunities(self):
        before = state({8: 1, 6: -1, 5: -1})
        move = Move([Step(8, 7), Step(7, 5)])
        row = observe_turn(before, apply(before, move), move, get_legal_moves(before, (1, 2), 1), 1)
        self.assertEqual(row["hits"], 1)
        self.assertEqual(row["max_available_hits"], 2)
        self.assertEqual(row["pip_progress"], 3)
        self.assertFalse(row["passed"])

    def test_progress_midpoint_distinguishes_rear_front_without_path_bias(self):
        before = state({20: 1, 8: 1, 24: -1})
        rear = outcome_features(before, apply(before, Move([Step(20, 18)])), 1)
        front = outcome_features(before, apply(before, Move([Step(8, 6)])), 1)
        self.assertEqual(rear["advancement_midpoint_distance"], 19)
        self.assertEqual(front["advancement_midpoint_distance"], 7)
        first = outcome_features(before, apply(before, Move([Step(20, 19), Step(19, 17)])), 1)
        second = outcome_features(before, apply(before, Move([Step(20, 18), Step(18, 17)])), 1)
        self.assertEqual(first, second)

    def test_safe_position_and_contact_metrics(self):
        before = state({8: 1, 7: 1, 1: -1})
        move = Move([Step(8, 6), Step(7, 6)])
        row = observe_turn(before, apply(before, move), move, get_legal_moves(before, (1, 2), 1), 1)
        self.assertEqual(row["threatened_blots"], 0)
        self.assertEqual(row["made_points"], 1)
        self.assertTrue(row["in_contact"])
        summary = summarize_turns([row])
        self.assertEqual(summary["mean_threatened_blots_in_contact"], 0)
        self.assertIsNone(summary["hit_when_available_rate"])

    def test_pass_and_pure_race_do_not_create_fake_opportunities(self):
        before = state({6: 2, 19: -1})
        row = observe_turn(before, before.copy(), None, [], 1)
        self.assertTrue(row["passed"])
        self.assertFalse(row["in_contact"])
        self.assertIsNone(row["advancement_choice_rank"])
        self.assertIsNone(summarize_turns([row])["mean_made_points_in_contact"])

    def test_terminal_board_is_not_contact_even_with_opposing_bar_checker(self):
        terminal = GameState()
        terminal.borne_off = {1: 15, -1: 14}
        terminal.bar[-1] = 1
        self.assertFalse(contact(terminal))

    def test_midpoints_are_symmetric_for_players(self):
        before = state({20: 1, 8: 1, 24: -1})
        after = apply(before, Move([Step(20, 18)]))
        def reflect(s):
            return GameState([-c for c in reversed(s.board)],
                             {p: s.bar[-p] for p in (1, -1)},
                             {p: s.borne_off[-p] for p in (1, -1)}, -1)
        self.assertEqual(outcome_features(before, after, 1),
                         outcome_features(reflect(before), reflect(after), -1))

    def test_game_summaries_keep_actual_candidate_and_opponent_separate(self):
        from evaluate_baseline import play_game, summarize
        record = play_game(0, 1120000, 1, 0, 10000, "random", "heuristic")
        rows = record["behavior_turns"]
        self.assertEqual(len(rows), record["turns"])
        self.assertEqual(sum(r["hits"] for r in rows), record["hits"])
        summary = summarize([record], "random")["gameplay_behavior"]
        self.assertEqual(summary["candidate"]["turns"], len(record["decision_seconds"]))
        self.assertEqual(summary["candidate"]["hits"] + summary["opponent"]["hits"], record["hits"])


if __name__ == "__main__":
    unittest.main()
