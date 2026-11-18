import unittest
from unittest.mock import patch

from agents.style_defaults import SELECTED_STYLE_STRENGTHS
from backgammon.rules import get_legal_moves
from backgammon.variants import RULE_VARIANTS
from evaluate_baseline import play_game
from monte_carlo.heuristic import DEFAULT_HEURISTIC
from rule_impact import compare_rule_outcomes
from run_tournament import PAIRS, jobs_for, run_game, summarize_results
from tests.test_rules import position


def cheap_move(agent, state, dice):
    moves = get_legal_moves(state, dice, state.current_player)
    return agent.rng.choice(moves) if moves else None


class TournamentTests(unittest.TestCase):
    def test_design_has_30_matchups_300_games_and_balanced_seats(self):
        settings = {'seed':1501000, 'games_per_matchup':10, 'variants':list(RULE_VARIANTS)}
        jobs = jobs_for(settings)
        self.assertEqual(len(PAIRS), 6)
        self.assertEqual(len(jobs), 300)
        self.assertEqual(len(set(jobs)), 300)
        for variant in RULE_VARIANTS:
            for a,b in PAIRS:
                selected = [j for j in jobs if j[:3] == (variant,a,b)]
                self.assertEqual(len(selected),10)
                self.assertEqual(sum(j[3]%2==0 for j in selected),5)
        self.assertEqual(len({j[4]+3*j[3] for j in jobs if j[0]=='standard'}),60)

    def test_worker_uses_selected_strengths_and_fixed_search(self):
        for a,b in PAIRS:
            with patch('run_tournament.play_game', return_value={}) as play:
                run_game(('relaxed-bar',a,b,1,1501000))
            self.assertEqual(play.call_args.args,(1,1501000,20,4,10000))
            self.assertEqual(play.call_args.kwargs,dict(opponent_kind=b,
                rollout_policy='heuristic',heuristic_config=DEFAULT_HEURISTIC,
                agent_kind=a,style_strength=SELECTED_STYLE_STRENGTHS[a],
                rule_variant='relaxed-bar',collect_rule_impact=True))

    def test_both_style_agents_receive_same_search_and_inherit_variant(self):
        seen=[]
        def tracked(agent,state,dice):
            seen.append((agent.style_name,agent.style_strength,agent.rollouts_per_move,
                         agent.depth,agent.rollout_policy,state.rule_variant))
            return cheap_move(agent,state,dice)
        with patch('agents.styled_agent.StyleBiasedAgent.choose_move',tracked):
            game=play_game(1,1500000,2,1,10000,opponent_kind='defensive',
                rollout_policy='heuristic',agent_kind='front-priority',
                rule_variant='no-doubles-bonus',collect_rule_impact=True)
        self.assertIn(('front-priority',.01,2,1,'heuristic','no-doubles-bonus'),seen)
        self.assertIn(('defensive',.5,2,1,'heuristic','no-doubles-bonus'),seen)
        self.assertEqual(game['mc_player'],-1)
        self.assertTrue(game['opponent_decision_seconds'])
        self.assertEqual(game['rule_impact']['turns_checked'],game['turns'])
        result={'variant':'no-doubles-bonus','agent_a':'front-priority','agent_b':'defensive',
                'game':game,'elapsed_seconds':1}
        summary=summarize_results([result],{'variants':list(RULE_VARIANTS),'games_per_matchup':10})
        self.assertEqual(summary['status'],'running')
        self.assertEqual(summary['expected_games'],300)
        self.assertEqual(sum(s['wins'] for s in summary['standings']),1)
        self.assertEqual(sum(s['gameplay_behavior']['turns'] for s in summary['standings']),game['turns'])

    def test_rule_impact_detects_each_variation_without_mutation(self):
        for variant,state,dice in [
            ('single-hit',position({8:1,7:-1,6:-1,5:-1}),(1,2)),
            ('relaxed-bar',position({8:1,24:-2,23:-2},bar=1),(1,2)),
            ('no-doubles-bonus',position({13:2}),(1,1)),
            ('eight-point-bearoff',position({8:1,3:1}),(3,1))]:
            with self.subTest(variant=variant):
                state.rule_variant=variant
                before=state.copy()
                moves=get_legal_moves(state,dice,1)
                self.assertTrue(compare_rule_outcomes(state,dice,1,moves))
                self.assertEqual(state,before)
        state=position({13:2})
        self.assertFalse(compare_rule_outcomes(state,(1,2),1,get_legal_moves(state,(1,2),1)))

    def test_empty_summary_does_not_claim_completion(self):
        s=summarize_results([],{'variants':list(RULE_VARIANTS),'games_per_matchup':10})
        self.assertEqual(s['completed_games'],0)
        self.assertEqual(s['expected_games'],300)
        self.assertEqual(s['status'],'running')
