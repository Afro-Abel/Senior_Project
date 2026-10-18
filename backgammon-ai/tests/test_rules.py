"""Rule fixtures use P1 coordinates, then mirror each case for P2."""
import unittest

from backgammon.move import BAR, OFF, Step
from backgammon.rules import apply_step, get_legal_moves, get_legal_steps
from backgammon.state import GameState


def position(points, bar=0):
    state = GameState()
    for point, count in points.items():
        state.board[point - 1] = count
    state.bar[1] = bar
    for player in (1, -1):
        state.borne_off[player] = 15 - state.checker_count(player)
    return state


def mirrored(state):
    return GameState(
        board=[-value for value in reversed(state.board)],
        bar={p: state.bar[-p] for p in (1, -1)},
        borne_off={p: state.borne_off[-p] for p in (1, -1)},
        current_player=-state.current_player,
    )


def mirror_step(step):
    return Step(BAR if step.start == BAR else 25 - step.start,
                OFF if step.end == OFF else 25 - step.end)


class RulesTests(unittest.TestCase):
    def check_moves(self, state, dice, expected):
        for player in (1, -1):
            with self.subTest(player=player, dice=dice):
                actual_state = state if player == 1 else mirrored(state)
                wanted = {tuple(steps) if player == 1 else
                          tuple(mirror_step(s) for s in steps)
                          for steps in expected}
                snapshot = actual_state.copy()
                moves = get_legal_moves(actual_state, dice, player)
                self.assertEqual({tuple(m.steps) for m in moves}, wanted)
                self.assertEqual(len(moves), len(wanted))
                self.assertEqual(actual_state, snapshot)

    def test_blocked_bar_forces_pass(self):
        self.check_moves(position({24: -2, 23: -2, 8: 14}, bar=1),
                         (1, 2), [])

    def test_bar_priority_and_higher_die(self):
        self.check_moves(position({22: -2}, bar=1), (1, 2),
                         [[Step(BAR, 23)]])

    def test_lower_die_allowed_when_higher_blocked(self):
        self.check_moves(position({23: -2, 22: -2}, bar=1),
                         (1, 2), [[Step(BAR, 24)]])

    def test_must_enter_all_bar_checkers_before_board_move(self):
        self.check_moves(position({8: 13}, bar=2), (1, 2),
                         [[Step(BAR, 24), Step(BAR, 23)],
                          [Step(BAR, 23), Step(BAR, 24)]])

    def test_order_that_uses_both_dice_is_required(self):
        self.check_moves(position({8: 1, 5: -2}), (3, 2),
                         [[Step(8, 6), Step(6, 3)]])

    def test_doubles_use_four_steps(self):
        self.check_moves(position({8: 1}), (1, 1),
                         [[Step(8, 7), Step(7, 6), Step(6, 5), Step(5, 4)]])

    def test_partially_playable_doubles(self):
        self.check_moves(position({8: 1, 5: -2}), (1, 1),
                         [[Step(8, 7), Step(7, 6)]])

    def test_last_checker_bearing_off_uses_higher_die(self):
        self.check_moves(position({3: 1}), (3, 6), [[Step(3, OFF)]])

    def test_can_enter_home_then_bear_off_same_turn(self):
        self.check_moves(position({7: 1}), (1, 6),
                         [[Step(7, 6), Step(6, OFF)],
                          [Step(7, 1), Step(1, OFF)]])

    def test_bearing_off_exact_oversized_and_not_ready(self):
        cases = [
            ({6: 1, 3: 1}, 3, [Step(6, 3), Step(3, OFF)]),
            ({5: 1, 3: 1}, 6, [Step(5, OFF)]),
            ({5: 1, 3: 1}, 2, [Step(3, 1), Step(5, 3)]),
            ({8: 1, 3: 1}, 6, [Step(8, 2)]),
        ]
        for points, die, expected in cases:
            for player in (1, -1):
                with self.subTest(points=points, die=die, player=player):
                    state = position(points)
                    if player == -1:
                        state = mirrored(state)
                        expected_steps = [mirror_step(s) for s in expected]
                    else:
                        expected_steps = expected
                    self.assertEqual(set(get_legal_steps(state, die, player)),
                                     set(expected_steps))

    def test_bar_entry_hits_and_copy_is_independent(self):
        for player in (1, -1):
            state = position({24: -1, 8: 14}, bar=1)
            step = Step(BAR, 24)
            if player == -1:
                state, step = mirrored(state), mirror_step(step)
            snapshot = state.copy()
            self.assertIn(step, get_legal_steps(state, 1, player))
            result = apply_step(state, step, player)
            self.assertEqual(state, snapshot)
            self.assertEqual(result.bar[player], 0)
            self.assertEqual(result.bar[-player], 1)
            self.assertEqual(result.board[step.end - 1], player)
            for p in (1, -1):
                self.assertEqual(result.checker_count(p), 15)


if __name__ == "__main__":
    unittest.main()
