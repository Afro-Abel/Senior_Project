"""Defensive style reduces direct blot threats and builds made points."""
from agents.styled_agent import StyleBiasedAgent
from monte_carlo.heuristic import contact_features


def made_points(state, player):
    """Count own protected points anywhere, including opponent-home anchors."""
    return sum(count * player >= 2 for count in state.board)


def contact_remains(state):
    if any(state.bar.values()):
        return True
    p1 = [point for point, count in enumerate(state.board, 1) if count > 0]
    p2 = [point for point, count in enumerate(state.board, 1) if count < 0]
    return bool(p1 and p2 and max(p1) > min(p2))


class DefensiveAgent(StyleBiasedAgent):
    style_name = "defensive"

    def raw_style_score(self, before, after, player):
        # Reuse the shared single-die threat proxy, not exact hitting odds.
        threats_before = contact_features(before, player)[2]
        threats_after = contact_features(after, player)[2]
        protection = (made_points(after, player) - made_points(before, player)
                      if contact_remains(after) else 0)
        # Gating the change (rather than subtracting gated potentials) avoids
        # penalizing the player merely for leaving contact and entering a race.
        return threats_before - threats_after + protection
