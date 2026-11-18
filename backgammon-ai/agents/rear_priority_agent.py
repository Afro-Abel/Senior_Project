"""Rear priority favors advancing checkers farther from bearing off."""
from agents.styled_agent import StyleBiasedAgent


def squared_distance_total(state, player):
    """Distances: bar=25, board=1..24 toward own exit, borne off=0."""
    return 25 ** 2 * state.bar[player] + sum(
        abs(count) * (point if player == 1 else 25 - point) ** 2
        for point, count in enumerate(state.board, 1) if count * player > 0)


class RearPriorityAgent(StyleBiasedAgent):
    style_name = "rear-priority"

    def raw_style_score(self, before, after, player):
        # Outcome-based potential reduction is independent of checker identity
        # and step order, preserving our candidate deduplication semantics.
        return squared_distance_total(before, player) - squared_distance_total(after, player)
