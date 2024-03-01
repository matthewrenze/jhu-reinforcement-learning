import pytest
import numpy as np
from agents.agent_factory import AgentFactory

@pytest.fixture
def factory():
    return AgentFactory()

@pytest.fixture
def state():
    return np.array([[0, 1], [2, 3]])

def test_get_agent_location(factory, state):
    agent_location = factory._get_agent_location(state)
    expected_location = (1, 0)
    assert agent_location == expected_location

def test_create_human_agent(factory, state):
    agent = factory.create("human", state)
    assert agent.__class__.__name__ == "HumanAgent"
    assert agent.location == (1, 0)

def test_create_random_agent(factory, state):
    factory = AgentFactory()
    agent = factory.create("random", state)
    assert agent.__class__.__name__ == "RandomAgent"
    assert agent.location == (1, 0)