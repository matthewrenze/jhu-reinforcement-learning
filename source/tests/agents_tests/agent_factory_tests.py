import numpy as np
import pytest
from unittest.mock import patch
from agents.agent_factory import AgentFactory
from tiles.test_tiles import TestTiles

@pytest.fixture
def setup():
    tiles = TestTiles.create([[0, 1], [2, 3]])
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3, "features":[0]}
    q_table = np.zeros((1, 2))
    feature_weights = np.zeros(len(hyperparameters["features"]))
    factory = AgentFactory()
    return tiles, hyperparameters, q_table, feature_weights, factory

@pytest.mark.parametrize("agent_name, agent_type", [
    ("human", "HumanAgent"),
    ("random", "RandomAgent"),
    ("sarsa", "SarsaAgent"),
    ("q_learning", "QLearningAgent"),
    ("approximate_q_learning", "ApproximateQLearningAgent"),
    ("deep_q_learning", "DeepQLearningAgent")
])
def test_create(setup, agent_name, agent_type):
    tiles, hyperparameters, q_table, feature_weights, factory = setup
    if agent_name == "approximate_q_learning": 
        factory._get_agent_model = feature_weights
    else:
        factory._get_agent_model = lambda x: q_table
    agent = factory.create(agent_name, tiles, hyperparameters)
    assert agent.__class__.__name__ == agent_type
    assert agent.location == (1, 0)
    assert agent.hyperparameters == hyperparameters

def test_get_agent_location(setup):
    tiles, hyperparameters, q_table, feature_weights, factory = setup
    agent_location = factory._get_agent_location(tiles)
    expected_location = (1, 0)
    assert agent_location == expected_location

