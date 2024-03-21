import pytest
import numpy as np
from agents.sarsa_agent import SarsaAgent
from agents.agent import Agent
from actions.action import Action
from states.state import State

@pytest.fixture()
def setup():
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3}
    q_table = np.zeros((20000, 5))
    agent = SarsaAgent((1, 2), hyperparameters, q_table)
    return hyperparameters, q_table, agent

def test_init(setup):
    hyperparameters, q_table, agent = setup
    assert agent.alpha == 0.1
    assert agent.gamma == 0.2
    assert agent.epsilon == 0.3
    assert agent.num_actions == 5
    assert agent.num_states == 20000
    assert agent.q_table.shape == (20000, 5)

def test_init_creates_q_table(setup):
    hyperparameters, q_table, _ = setup
    agent = SarsaAgent((1, 2), hyperparameters, None)
    assert agent.q_table.shape == (20000, 5)



