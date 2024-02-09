import numpy as np
from unittest.mock import patch
from agents.human_agent import HumanAgent
from agents.actions import Action

@patch("agents.human_agent.get_key", return_value="k")
def test_select_action(mock_get_key):
    agent = HumanAgent()
    state = np.zeros((2, 2))
    action = agent.select_action(state)
    assert action == Action.DOWN.value