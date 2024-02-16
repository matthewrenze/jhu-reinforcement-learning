import pytest
import numpy as np
from agents.human_agent import HumanAgent
from agents.actions import Action

@pytest.mark.parametrize("key, expected_action", [
    ("i", Action.UP.value),
    ("l", Action.RIGHT.value),
    ("k", Action.DOWN.value),
    ("j", Action.LEFT.value),
    ("", Action.NONE.value)
])
def test_select_action(key, expected_action):
    agent = HumanAgent()
    agent._get_key = lambda: key
    state = np.zeros((2, 2))
    actual_action = agent.select_action(state)
    assert expected_action == actual_action

def test_select_invalid_key_returns_no_action():
    agent = HumanAgent()
    agent._get_key = lambda: "a"
    state = np.zeros((2, 2))
    expected_action = Action.NONE.value
    actual_action = agent.select_action(state)
    assert expected_action == actual_action

