"""Style-independent observations of actual turns, not simulated rollouts."""
from statistics import mean

from backgammon.rules import apply_step
from monte_carlo.heuristic import contact_features, pip_count


def contact(state):
    if any(count == 15 for count in state.borne_off.values()):
        return False
    p1 = [i for i, c in enumerate(state.board) if c > 0]
    p2 = [i for i, c in enumerate(state.board) if c < 0]
    return bool(any(state.bar.values()) or (p1 and p2 and max(p1) > min(p2)))


def distance_squared(state, player):
    return 625 * state.bar[player] + sum(
        abs(c) * (i if player == 1 else 25 - i) ** 2
        for i, c in enumerate(state.board, 1) if c * player > 0)


def outcome_features(before, after, player):
    progress = pip_count(before, player) - pip_count(after, player)
    reduction = distance_squared(before, player) - distance_squared(after, player)
    return {"hits": after.bar[-player] - before.bar[-player],
            "pip_progress": progress,
            "squared_distance_reduction": reduction,
            # Mean distance along actual checker travel, weighted by pips.
            # Outcome-based: equivalent sequences have identical measurements.
            "advancement_midpoint_distance": reduction / (2 * progress) if progress else None,
            "borne_off": after.borne_off[player] - before.borne_off[player],
            "blots": sum(c == player for c in after.board),
            "threatened_blots": contact_features(after, player)[2],
            "made_points": sum(c * player >= 2 for c in after.board),
            "in_contact": contact(after)}


def observe_turn(before, after, move, legal_moves, player):
    chosen = outcome_features(before, after, player)
    options = []
    for option in legal_moves:
        hypothetical = before.copy()
        for step in option.steps:
            hypothetical = apply_step(hypothetical, step, player)
        options.append(outcome_features(before, hypothetical, player))
    available = [o["advancement_midpoint_distance"] for o in options
                 if o["advancement_midpoint_distance"] is not None]
    low, high = (min(available), max(available)) if available else (0, 0)
    value = chosen["advancement_midpoint_distance"]
    chosen.update({"passed": move is None,
                   "max_available_hits": max((o["hits"] for o in options), default=0),
                   "advancement_choice_rank": ((value - low) / (high - low)
                                               if high > low and value is not None else None)})
    return chosen


def summarize_turns(turns):
    if not turns:
        return None
    contact_turns = [t for t in turns if t["in_contact"]]
    opportunities = [t for t in turns if t["max_available_hits"] > 0]
    ranks = [t["advancement_choice_rank"] for t in turns if t["advancement_choice_rank"] is not None]
    progress = sum(t["pip_progress"] for t in turns)
    return {"turns": len(turns), "passes": sum(t["passed"] for t in turns),
            "hits": sum(t["hits"] for t in turns),
            "hits_per_turn": sum(t["hits"] for t in turns) / len(turns),
            "hitting_opportunity_turns": len(opportunities),
            "hit_when_available_rate": (mean(t["hits"] > 0 for t in opportunities)
                                        if opportunities else None),
            "mean_post_turn_blots": mean(t["blots"] for t in turns),
            "mean_post_turn_threatened_blots": mean(t["threatened_blots"] for t in turns),
            "mean_post_turn_made_points": mean(t["made_points"] for t in turns),
            "contact_turns": len(contact_turns),
            "mean_threatened_blots_in_contact": (mean(t["threatened_blots"] for t in contact_turns)
                                                 if contact_turns else None),
            "mean_made_points_in_contact": (mean(t["made_points"] for t in contact_turns)
                                             if contact_turns else None),
            "pip_progress": progress,
            "mean_advancement_midpoint_distance": (sum(t["squared_distance_reduction"] for t in turns)
                                                    / (2 * progress) if progress else None),
            "advancement_choice_turns": len(ranks),
            "mean_advancement_choice_rank": mean(ranks) if ranks else None,
            "checkers_borne_off": sum(t["borne_off"] for t in turns)}
