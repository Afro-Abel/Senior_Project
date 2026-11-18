from dataclasses import replace
from math import tanh
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from agents.monte_carlo_agent import MonteCarloAgent
from backgammon.state import GameState
from monte_carlo.heuristic import DEFAULT_HEURISTIC, HeuristicConfig, evaluate_position, load_heuristic_config
from monte_carlo.rollout import rollout


class ConfigTests(unittest.TestCase):
    def test_partial_overrides_and_bad_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"borne_off": 0}')
            config = load_heuristic_config(path)
            self.assertEqual(config, replace(DEFAULT_HEURISTIC, borne_off=0))
            for text in ('{"born_off": 0}', '[]'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    load_heuristic_config(path)

    def test_invalid_parameters(self):
        for config in ({"scale": 0}, {"scale": -1}, {"bar": -1},
                       {"borne_off": True}, {"scale": float("nan")},
                       {"home_points": float("inf")}, {"bar": "8"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                HeuristicConfig(**config)

    def test_ablation_and_scale_match_expected_race_scores(self):
        state = GameState()
        state.board[2], state.board[21] = 1, -2
        state.borne_off = {1: 14, -1: 13}
        # Opponent has six remaining pips versus our three, plus one off advantage.
        self.assertAlmostEqual(evaluate_position(state, 1), .99 * tanh(7 / 50))
        no_off = replace(DEFAULT_HEURISTIC, borne_off=0)
        self.assertAlmostEqual(evaluate_position(state, 1, no_off), .99 * tanh(3 / 50))
        self.assertAlmostEqual(evaluate_position(state, 1, replace(no_off, scale=25)),
                               .99 * tanh(3 / 25))
        self.assertAlmostEqual(evaluate_position(state, -1, no_off),
                               -evaluate_position(state, 1, no_off))
        state.board[2] = 0
        state.borne_off[1] = 15
        self.assertEqual(evaluate_position(state, 1, no_off), 1)

    def test_custom_config_reaches_guided_choices_and_cutoff(self):
        config = replace(DEFAULT_HEURISTIC, borne_off=0, scale=100)
        state = GameState.initial_state()
        with patch("monte_carlo.rollout.evaluate_position", wraps=evaluate_position) as score:
            rollout(state, random.Random(3), 2, 1, "heuristic", config)
        self.assertGreater(score.call_count, 2)
        self.assertTrue(all(call.args[2] is config for call in score.call_args_list))
        self.assertEqual(score.call_args.args[1], 1)

    def test_agent_forwards_config_without_changing_defaults(self):
        config = replace(DEFAULT_HEURISTIC, borne_off=0)
        with patch("agents.monte_carlo_agent.rollout", return_value=0) as simulate:
            MonteCarloAgent(1, heuristic_config=config).evaluate_moves(GameState.initial_state(), (2, 6))
        self.assertTrue(all(call.args[-1] is config for call in simulate.call_args_list))
        self.assertEqual(MonteCarloAgent().heuristic_config, DEFAULT_HEURISTIC)


if __name__ == "__main__":
    unittest.main()
