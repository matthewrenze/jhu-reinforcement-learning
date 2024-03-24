from agents.agent import Agent
from actions.action import Action

def test_init():
    hyperparameters = {"key": 0.1}
    agent = Agent((1, 2), hyperparameters)
    assert agent.location == (1, 2)
    assert agent.orientation == Action.RIGHT
    assert agent.hyperparameters == hyperparameters
