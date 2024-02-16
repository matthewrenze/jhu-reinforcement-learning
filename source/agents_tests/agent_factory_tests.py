import pytest

from agents.agent_factory import AgentFactory

@pytest.fixture
def factory():
    return AgentFactory()

def test_create_human_agent(factory):
    agent = factory.create("human")
    assert agent.__class__.__name__ == "HumanAgent"

def test_create_random_agent(factory):
    factory = AgentFactory()
    agent = factory.create("random")
    assert agent.__class__.__name__ == "RandomAgent"