"""Aggressive style prefers actions that hit more opposing checkers."""
from agents.styled_agent import StyleBiasedAgent


class AggressiveAgent(StyleBiasedAgent):
    style_name = "aggressive"

    def raw_style_score(self, before, after, player):
        # The opponent cannot re-enter during our complete turn, so their bar
        # increase counts precisely the checkers hit by this action. Existing
        # bar checkers are not rewarded again.
        return after.bar[-player] - before.bar[-player]
