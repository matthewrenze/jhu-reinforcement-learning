from unittest.mock import Mock
from agents.random_agent import RandomAgent
from actions.action import Action

def test_select_action():
    agent = RandomAgent((0, 0))
    agent._get_random_action = lambda: Action.UP
    state = Mock()
    action = agent.select_action(state)
    assert action == Action.UP