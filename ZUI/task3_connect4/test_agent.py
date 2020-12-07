#!/usr/bin/env python3
from utils import *
import math
import numpy as np  # type: ignore
import pytest  # type: ignore


class TestDeterministicAgent:
    def test_tree_fast_1(self):
        from agent import DeterministicAgent as Agent
        value, _, estimated_nodes_count = tree_game_estimate(Agent, 1, 4)
        assert value == -63

    def test_tree_fast_2(self):
        from agent import DeterministicAgent as Agent
        value, _, estimated_nodes_count = tree_game_estimate(Agent, 2, 4)
        assert value == 69

    def test_tree_fast_3(self):
        from agent import DeterministicAgent as Agent
        value, _, estimated_nodes_count = tree_game_estimate(Agent, 3, 5)
        assert value == 36

    def test_tree_fast_4(self):
        from agent import DeterministicAgent as Agent
        value, _, estimated_nodes_count = tree_game_estimate(Agent, 4, 5)
        assert value == 43

    @pytest.mark.slow
    def test_tree_game_1(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = tree_game_estimate(Agent, 1, 9)
        assert value == -61
        assert expanded_nodes_count <= 20000

    @pytest.mark.slow
    def test_tree_game_2(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = tree_game_estimate(Agent, 2, 9)
        assert value == 75
        assert expanded_nodes_count <= 16000

    @pytest.mark.slow
    def test_connect4_1(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = connect4_estimate(Agent, 1, 7, 6)
        assert value == 2900
        assert expanded_nodes_count <= 12000

    @pytest.mark.slow
    def test_connect4_2(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = connect4_estimate(Agent, 2, 7, 6)
        assert value == 3600
        assert expanded_nodes_count <= 12000

    @pytest.mark.slow
    def test_connect4_3(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = connect4_estimate(Agent, 1, 9, 6)
        assert value == 3800
        assert expanded_nodes_count <= 160000

    @pytest.mark.slow
    def test_connect4_4(self):
        from agent import DeterministicAgent as Agent
        value, expanded_nodes_count, estimated_nodes_count = connect4_estimate(Agent, 2, 9, 6)
        assert value == 3500
        assert expanded_nodes_count <= 200000


class TestMonteCarloTreeSearch:
    def test_multi_armed_1(self):
        from agent import MonteCarloAgent as Agent
        from testing_games import MultiArmedBanditGame
        simulations = 8000
        game = MultiArmedBanditGame(2, simulations, 1, scale=0.01)
        agent = Agent(simulations, 2)
        r = agent.mcts(game.clone())  # !
        value = max(game._means)

        assert r.visit_count == simulations
        assert r.value() < value
        assert abs(value - r.value()) <= 0.09

    def test_multi_armed_2(self):
        from agent import MonteCarloAgent as Agent
        from testing_games import MultiArmedBanditGame
        simulations = 8000
        game = MultiArmedBanditGame(3, simulations, 1, scale=0.01)
        agent = Agent(simulations, 2)
        r = agent.mcts(game.clone())
        value = min(game._means)

        assert r.visit_count == simulations
        assert r.value() > value
        assert abs(value - r.value()) <= 0.09

    def test_multi_armed_3(self):
        from agent import MonteCarloAgent as Agent
        from testing_games import MultiArmedBanditGame
        simulations = 8000
        game = MultiArmedBanditGame(50, simulations, 1, scale=0.1)
        agent = Agent(simulations, 2)
        r = agent.mcts(game.clone())
        value = max(game._means)

        assert r.visit_count == simulations
        assert r.value() < value
        assert abs(value - r.value()) <= 0.09

    def test_multi_armed_4(self):
        from agent import MonteCarloAgent as Agent
        from testing_games import MultiArmedBanditGame
        simulations = 8000
        game = MultiArmedBanditGame(51, simulations, 1, scale=0.1)
        agent = Agent(simulations, 2)
        r = agent.mcts(game.clone())
        value = min(game._means)

        assert r.visit_count == simulations
        assert r.value() > value
        assert abs(value - r.value()) <= 0.09

    @pytest.mark.slow
    def test_connect4_1(self):
        from agent import MonteCarloAgent as Agent
        zstat = ztest(connect4_mean_and_std(Agent, 1, 18, simulations=2000), (-0.5683, 0.0540))

        assert abs(zstat) < ZTEST_LEVEL, "likely incorrect"

    @pytest.mark.slow
    def test_connect4_2(self):
        from agent import MonteCarloAgent as Agent
        zstat = ztest(connect4_mean_and_std(Agent, 1, 16, simulations=2000), (-0.5029, 0.0265))
        assert abs(zstat) < ZTEST_LEVEL, "likely incorrect"
