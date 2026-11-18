"""Front priority favors advancing checkers closer to bearing off."""
from agents.rear_priority_agent import squared_distance_total
from agents.styled_agent import StyleBiasedAgent
from monte_carlo.heuristic import pip_count


class FrontPriorityAgent(StyleBiasedAgent):
    style_name = "front-priority"

    def raw_style_score(self, before, after, player):
        # Potential F(d)=52*d-d*d is increasing over d=0..25, with a
        # larger reduction per pip near the exit. Use complete outcomes so
        # equivalent step sequences get the same preference.
        progress = pip_count(before, player) - pip_count(after, player)
        rear_progress = squared_distance_total(before, player) - squared_distance_total(after, player)
        return 52 * progress - rear_progress
