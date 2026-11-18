"""Compare distinct legal resulting positions under active and standard rules.

This measures available-action changes on visited states, not causal win effects.
It does not simulate counterfactual game paths or include rollout positions.
"""
from backgammon.rules import apply_step, get_legal_moves


def outcome_keys(state, moves, player):
    keys = set()
    for move in moves:
        after = state.copy()
        for step in move.steps:
            after = apply_step(after, step, player)
        keys.add((tuple(after.board), tuple(after.bar[p] for p in (1, -1)),
                  tuple(after.borne_off[p] for p in (1, -1))))
    return keys


def compare_rule_outcomes(state, dice, player, active_moves):
    if state.rule_variant == "standard":
        return False
    reference = state.copy()
    reference.rule_variant = "standard"
    standard_moves = get_legal_moves(reference, dice, player)
    return outcome_keys(state, active_moves, player) != outcome_keys(reference, standard_moves, player)
