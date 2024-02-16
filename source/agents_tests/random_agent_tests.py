import numpy as np
from unittest.mock import patch
from agents.random_agent import RandomAgent
from agents.actions import Action

def test_select_action():
    agent = RandomAgent()
    agent._get_random_action = lambda: Action.UP.value
    state = np.zeros((2, 2))
    action = agent.select_action(state)
    assert action == Action.UP.value