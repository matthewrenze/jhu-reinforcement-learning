import numpy as np
import pytest
from unittest.mock import patch
from agents.agent_factory import AgentFactory
from tiles.test_tiles import TestTiles

@pytest.fixture
def setup():
    tiles = TestTiles.create([[0, 1], [2, 3]])
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3}
    q_table = np.zeros((1, 2))
    factory = AgentFactory()
    return tiles, hyperparameters, q_table, factory

@pytest.mark.parametrize("agent_name, agent_type", [
    ("human", "HumanAgent"),
    ("random", "RandomAgent"),
    ("sarsa", "SarsaAgent")])
def test_create(setup, agent_name, agent_type):
    tiles, hyperparameters, q_table, factory = setup
    factory._get_agent_q_table = lambda x: q_table
    agent = factory.create(agent_name, tiles, hyperparameters)
    assert agent.__class__.__name__ == agent_type
    assert agent.location == (1, 0)
    assert agent.hyperparameters == hyperparameters

@patch("numpy.savetxt")
def test_save(mock_savetxt, setup):
    tiles, hyperparameters, q_table, factory = setup
    factory.save("sarsa", q_table)
    mock_savetxt.assert_called_with("../q_tables/sarsa.csv", q_table, delimiter=",")

def test_get_agent_location(setup):
    tiles, hyperparameters, q_table, factory = setup
    agent_location = factory._get_agent_location(tiles)
    expected_location = (1, 0)
    assert agent_location == expected_location

@patch("os.path.exists", return_value=True)
@patch("numpy.loadtxt", return_value=np.zeros((1, 2)))
def test_get_agent_q_table(mock_loadtxt, mock_exists, setup):
    tiles, hyperparameters, q_table, factory = setup
    expected_q_table = np.zeros((1, 2))
    q_table = factory._get_agent_q_table("sarsa")
    assert q_table.shape == expected_q_table.shape
    mock_exists.assert_called_with("../q_tables/sarsa.csv")
    mock_loadtxt.assert_called_with("../q_tables/sarsa.csv", delimiter=",")

