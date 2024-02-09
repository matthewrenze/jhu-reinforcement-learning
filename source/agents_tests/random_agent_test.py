import numpy as np
from unittest.mock import patch
from agents.random_agent import RandomAgent
from agents.actions import Action

@patch("agents.random_agent.get_random_action", return_value=Action.UP.value)
def test_select_action(mock_get_random_action):
    agent = RandomAgent()
    state = np.zeros((2, 2))
    action = agent.select_action(state)
    assert action == Action.UP.value