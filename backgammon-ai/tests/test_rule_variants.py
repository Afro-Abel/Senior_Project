import random
import unittest
from unittest.mock import patch

from backgammon.game import BackgammonGame
from backgammon.move import Move, Step, BAR
from backgammon.rules import apply_step, get_legal_moves
from backgammon.state import GameState
from monte_carlo.rollout import rollout
from validate_random_games import dice_usage_checker
from tests.test_rules import position, mirrored


def outcome(state, move, player):
    for step in move.steps:
        state = apply_step(state, step, player)
    return state


class SingleHitTests(unittest.TestCase):
    def both_players(self, state):
        for player, current in [(1, state), (-1, mirrored(state))]:
            current.rule_variant = "single-hit"
            current.current_player = player
            yield player, current

    def test_variant_validation_and_copy(self):
        with self.assertRaises(ValueError):
            GameState(rule_variant="unknown")
        game = BackgammonGame(rule_variant="single-hit")
        self.assertEqual(game.state.copy().rule_variant, "single-hit")
        self.assertEqual(GameState.initial_state().rule_variant, "standard")

    def test_single_hit_limits_sequences_not_individual_steps(self):
        original = position({8: 1, 7: -1, 5: -1})
        standard = get_legal_moves(original, (1, 2), 1)
        self.assertTrue(any(outcome(original, m, 1).bar[-1] == 2 for m in standard))
        for player, state in self.both_players(original):
            moves = get_legal_moves(state, (1, 2), player)
            self.assertTrue(moves)
            optimum, check = dice_usage_checker(state, (1, 2), player)
            self.assertEqual(optimum[0], 2)
            for move in moves:
                check(move)
                self.assertLessEqual(outcome(state, move, player).bar[-player], 1)

    def test_constraint_changes_maximum_dice_and_higher_die(self):
        # Both orders require two hits; under the variation only one die is usable.
        original = position({8: 1, 7: -1, 6: -1, 5: -1})
        for player, state in self.both_players(original):
            moves = get_legal_moves(state, (1, 2), player)
            self.assertTrue(moves)
            optimum, check = dice_usage_checker(state, (1, 2), player)
            self.assertEqual(optimum, (1, 2))
            for move in moves:
                self.assertEqual(len(move.steps), 1)
                check(move)

    def test_doubles_and_bar_entry_count_hits(self):
        for original, dice in [(position({8: 1, 7: -1, 6: -1}), (1, 1)),
                               (position({24: -1, 22: -1}, bar=1), (1, 2))]:
            for player, state in self.both_players(original):
                moves = get_legal_moves(state, dice, player)
                self.assertTrue(moves)
                _, check = dice_usage_checker(state, dice, player)
                for move in moves:
                    check(move)
                    self.assertLessEqual(outcome(state, move, player).bar[-player], 1)

    def test_application_rejects_second_hit_atomically(self):
        game = BackgammonGame(rule_variant="single-hit")
        game.state = position({8: 1, 7: -1, 5: -1})
        game.state.rule_variant = "single-hit"
        before = game.state.copy()
        with self.assertRaises(ValueError):
            game.apply_move(Move([Step(8, 7), Step(7, 5)]))
        self.assertEqual(game.state, before)
        game.apply_move(Move([Step(8, 7)]))
        game.state.current_player = 1
        # A new apply_move call is a new turn; the hit counter must reset.
        game.apply_move(Move([Step(7, 5)]))
        self.assertEqual(game.state.bar[-1], 2)

    def test_rollout_inherits_rule_and_does_not_mutate_input(self):
        state = GameState.initial_state("single-hit")
        before = state.copy()
        observed = []
        def legal(current, dice, player):
            observed.append(current.rule_variant)
            moves = get_legal_moves(current, dice, player)
            for move in moves:
                self.assertLessEqual(outcome(current, move, player).bar[-player] - current.bar[-player], 1)
            return moves
        with patch("monte_carlo.rollout.get_legal_moves", side_effect=legal):
            rollout(state, random.Random(12), 8, 1, "heuristic")
        self.assertEqual(observed, ["single-hit"] * 8)
        self.assertEqual(state, before)

    def test_standard_and_variant_match_when_no_hits_available(self):
        state = position({6: 3, 5: 2})
        standard = get_legal_moves(state, (2, 2), 1)
        state.rule_variant = "single-hit"
        self.assertEqual(get_legal_moves(state, (2, 2), 1), standard)


class RelaxedBarTests(unittest.TestCase):
    def sides(self, original):
        for player, state in [(1, original), (-1, mirrored(original))]:
            state.rule_variant = "relaxed-bar"
            state.current_player = player
            yield player, state

    def test_entry_and_board_moves_are_both_available(self):
        from backgammon.rules import get_legal_steps
        original = position({8: 1}, bar=1)
        for player, state in self.sides(original):
            steps = get_legal_steps(state, 1, player)
            self.assertTrue(any(s.start == BAR for s in steps))
            self.assertTrue(any(s.start != BAR for s in steps))
            state.rule_variant = "standard"
            self.assertTrue(all(s.start == BAR for s in get_legal_steps(state, 1, player)))

    def test_blocked_entry_no_longer_forces_pass_and_uses_both_dice(self):
        original = position({8: 1, 24: -2, 23: -2}, bar=1)
        for player, state in self.sides(original):
            snapshot = state.copy()
            moves = get_legal_moves(state, (1, 2), player)
            self.assertTrue(moves)
            optimum, check = dice_usage_checker(state, (1, 2), player)
            self.assertEqual(optimum[0], 2)
            for move in moves:
                check(move)
                after = outcome(state, move, player)
                self.assertEqual(after.bar[player], 1)
            self.assertEqual(state, snapshot)
            state.rule_variant = "standard"
            self.assertEqual(get_legal_moves(state, (1, 2), player), [])

    def test_bearing_off_still_requires_empty_bar(self):
        from backgammon.rules import can_bear_off, get_legal_steps
        from backgammon.move import OFF
        for player, state in self.sides(position({1: 1}, bar=1)):
            self.assertFalse(can_bear_off(state, player))
            self.assertFalse(any(s.end == OFF for s in get_legal_steps(state, 1, player)))

    def test_doubles_allow_board_movement_with_blocked_bar(self):
        for player, state in self.sides(position({8: 1, 24: -2}, bar=1)):
            moves = get_legal_moves(state, (1, 1), player)
            self.assertTrue(moves)
            _, check = dice_usage_checker(state, (1, 1), player)
            for move in moves:
                self.assertEqual(len(move.steps), 4)
                check(move)

    def test_exposure_includes_opponent_board_attack_with_bar(self):
        from monte_carlo.heuristic import contact_features
        state = position({10: 1, 8: -1})
        state.bar[-1] = 1
        state.borne_off[-1] -= 1
        self.assertEqual(contact_features(state, 1)[2], 0)
        for player, current in self.sides(state):
            self.assertEqual(contact_features(current, player)[2], 1)

    def test_rollout_inherits_relaxed_bar(self):
        state = GameState.initial_state("relaxed-bar")
        snapshot = state.copy()
        seen = []
        def legal(current, dice, player):
            seen.append(current.rule_variant)
            return get_legal_moves(current, dice, player)
        with patch("monte_carlo.rollout.get_legal_moves", side_effect=legal):
            rollout(state, random.Random(8), 4, 1, "heuristic")
        self.assertEqual(seen, ["relaxed-bar"] * 4)
        self.assertEqual(state, snapshot)

    def test_without_bar_same_moves_as_standard(self):
        state = GameState.initial_state()
        standard = get_legal_moves(state, (2, 6), 1)
        state.rule_variant = "relaxed-bar"
        self.assertEqual(state.copy().rule_variant, "relaxed-bar")
        self.assertEqual(get_legal_moves(state, (2, 6), 1), standard)


class NoDoublesBonusTests(unittest.TestCase):
    def sides(self, original):
        for player, state in [(1, original), (-1, mirrored(original))]:
            state.rule_variant = "no-doubles-bonus"
            state.current_player = player
            yield player, state

    def test_two_uses_same_checker_or_two_checkers(self):
        original = position({13: 2})
        for player, state in self.sides(original):
            moves = get_legal_moves(state, (4, 4), player)
            self.assertTrue(moves)
            optimum, check = dice_usage_checker(state, (4, 4), player)
            self.assertEqual(optimum, (2, 8))
            for move in moves:
                check(move)
                self.assertEqual(len(move.steps), 2)
            starts = [(m.steps[0].start, m.steps[1].start) for m in moves]
            self.assertTrue(any(a == b for a, b in starts))
            self.assertTrue(any(a != b for a, b in starts))
            state.rule_variant = "standard"
            self.assertTrue(all(len(m.steps) == 4 for m in get_legal_moves(state, (4, 4), player)))

    def test_blocked_intermediate_point_cannot_be_jumped(self):
        for player, state in self.sides(position({13: 1, 9: -2})):
            self.assertEqual(get_legal_moves(state, (4, 4), player), [])
            self.assertEqual(dice_usage_checker(state, (4, 4), player)[0], (0, 0))

    def test_only_one_use_possible(self):
        for player, state in self.sides(position({13: 1, 5: -2})):
            moves = get_legal_moves(state, (4, 4), player)
            self.assertTrue(moves)
            optimum, check = dice_usage_checker(state, (4, 4), player)
            self.assertEqual(optimum, (1, 4))
            for move in moves:
                check(move)
                self.assertEqual(len(move.steps), 1)

    def test_bar_priority_and_bearing_off_unchanged(self):
        from backgammon.move import OFF
        for player, state in self.sides(position({13: 1}, bar=2)):
            moves = get_legal_moves(state, (4, 4), player)
            self.assertTrue(moves)
            for move in moves:
                self.assertEqual([s.start for s in move.steps], [BAR, BAR])
                self.assertEqual(outcome(state, move, player).bar[player], 0)
        for player, state in self.sides(position({4: 2})):
            moves = get_legal_moves(state, (4, 4), player)
            _, check = dice_usage_checker(state, (4, 4), player)
            self.assertTrue(moves)
            for move in moves:
                check(move)
                self.assertEqual([s.end for s in move.steps], [OFF, OFF])
                self.assertEqual(outcome(state, move, player).borne_off[player], 15)

    def test_non_doubles_unchanged_and_copy_preserves_variant(self):
        state = GameState.initial_state()
        standard = get_legal_moves(state, (2, 6), 1)
        state.rule_variant = "no-doubles-bonus"
        self.assertEqual(state.copy().rule_variant, "no-doubles-bonus")
        self.assertEqual(get_legal_moves(state, (2, 6), 1), standard)

    def test_rollout_uses_two_steps_on_doubles(self):
        state = position({13: 2, 1: -2})
        state.rule_variant = "no-doubles-bonus"
        snapshot = state.copy()
        observed = []
        def legal(current, dice, player):
            self.assertEqual(current.rule_variant, "no-doubles-bonus")
            moves = get_legal_moves(current, dice, player)
            observed.extend(moves)
            return moves
        with patch.object(BackgammonGame, "roll_dice", return_value=(4, 4)), \
             patch("monte_carlo.rollout.get_legal_moves", side_effect=legal):
            rollout(state, random.Random(2), 1, 1, "heuristic")
        self.assertTrue(observed)
        self.assertTrue(all(len(m.steps) == 2 for m in observed))
        self.assertEqual(state, snapshot)


class EightPointBearoffTests(unittest.TestCase):
    def sides(self, original):
        for player, state in [(1, original), (-1, mirrored(original))]:
            state.rule_variant = "eight-point-bearoff"
            state.current_player = player
            yield player, state

    def test_zone_boundary_and_bar(self):
        from backgammon.rules import can_bear_off
        for point in (7, 8, 9):
            for player, state in self.sides(position({point: 1})):
                self.assertEqual(can_bear_off(state, player), point <= 8)
                state.rule_variant = "standard"
                self.assertFalse(can_bear_off(state, player))
        for player, state in self.sides(position({1: 1}, bar=1)):
            self.assertFalse(can_bear_off(state, player))

    def test_exact_roll_allowed_while_rear_checker_at_eight(self):
        from backgammon.rules import get_legal_steps
        from backgammon.move import OFF
        for player, state in self.sides(position({8: 1, 3: 1})):
            ownpoint = 3 if player == 1 else 22
            self.assertIn(Step(ownpoint, OFF), get_legal_steps(state, 3, player))
            state.rule_variant = "standard"
            self.assertNotIn(Step(ownpoint, OFF), get_legal_steps(state, 3, player))

    def test_oversized_roll_blocked_by_checker_at_seven_or_eight(self):
        from backgammon.rules import get_legal_steps
        from backgammon.move import OFF
        for rear in (7, 8):
            for player, state in self.sides(position({rear: 1, 3: 1})):
                ownpoint = 3 if player == 1 else 22
                self.assertNotIn(Step(ownpoint, OFF), get_legal_steps(state, 6, player))
        for player, state in self.sides(position({3: 1})):
            ownpoint = 3 if player == 1 else 22
            self.assertIn(Step(ownpoint, OFF), get_legal_steps(state, 6, player))

    def test_seven_and_eight_require_multiple_legal_steps(self):
        from backgammon.rules import get_legal_steps
        from backgammon.move import OFF
        for player, state in self.sides(position({8: 1})):
            self.assertFalse(any(s.end == OFF for s in get_legal_steps(state, 6, player)))
            moves = get_legal_moves(state, (6, 2), player)
            self.assertTrue(moves)
            _, check = dice_usage_checker(state, (6, 2), player)
            for move in moves:
                check(move)
                self.assertEqual(len(move.steps), 2)
                self.assertEqual(outcome(state, move, player).borne_off[player], 15)

    def test_can_enter_zone_then_bear_off_same_turn(self):
        from backgammon.move import OFF
        for player, state in self.sides(position({9: 1, 2: 1})):
            moves = get_legal_moves(state, (1, 2), player)
            self.assertTrue(moves)
            _, check = dice_usage_checker(state, (1, 2), player)
            for move in moves:
                check(move)
            self.assertTrue(any(m.steps[1].end == OFF for m in moves))

    def test_rollout_inherits_zone_and_preserves_input(self):
        state = position({8: 1, 3: 1, 1: -2})
        state.rule_variant = "eight-point-bearoff"
        snapshot = state.copy()
        seen = []
        def legal(current, dice, player):
            seen.append(current.rule_variant)
            return get_legal_moves(current, dice, player)
        with patch.object(BackgammonGame, "roll_dice", return_value=(3, 1)), \
             patch("monte_carlo.rollout.get_legal_moves", side_effect=legal):
            rollout(state, random.Random(2), 1, 1, "heuristic")
        self.assertEqual(seen, ["eight-point-bearoff"])
        self.assertEqual(state, snapshot)
