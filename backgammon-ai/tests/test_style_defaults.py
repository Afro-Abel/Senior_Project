import unittest
from agents.rear_priority_agent import RearPriorityAgent
from agents.front_priority_agent import FrontPriorityAgent
from agents.aggressive_agent import AggressiveAgent
from agents.defensive_agent import DefensiveAgent

class SelectedDefaultsTests(unittest.TestCase):
    def test_selected_defaults_and_explicit_overrides(self):
        for cls, strength in [(RearPriorityAgent,.05), (FrontPriorityAgent,.01),
                              (AggressiveAgent,.05), (DefensiveAgent,.5)]:
            self.assertEqual(cls().style_strength, strength)
            self.assertEqual(cls(style_strength=0).style_strength, 0)
            self.assertEqual(cls(style_strength=.123).style_strength, .123)
