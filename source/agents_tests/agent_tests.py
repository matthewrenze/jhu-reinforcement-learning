from agents.agent import Agent
from actions.action import Action

def test_init():
    agent = Agent((1, 2))
    assert agent.location == (1, 2)
    assert agent.orientation == Action.RIGHT
