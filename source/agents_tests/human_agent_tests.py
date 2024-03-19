import pytest
from unittest.mock import Mock
from agents.human_agent import HumanAgent
from actions.action import Action

# NOTE: If no action is selected, then we use previous orientation
# NOTE: In this case, we use the default orientation, which is Action.RIGHT

@pytest.mark.parametrize("key, expected_action, expected_orientation", [
    ("i", Action.UP, Action.UP),
    ("l", Action.RIGHT, Action.RIGHT),
    ("k", Action.DOWN, Action.DOWN),
    ("j", Action.LEFT, Action.LEFT),
    ("", Action.NONE, Action.RIGHT)
])
def test_select_action(key, expected_action, expected_orientation):
    agent = HumanAgent((0, 0))
    agent._get_key = lambda: key
    state = Mock()
    actual_action = agent.select_action(state)
    assert expected_action == actual_action
    assert agent.orientation == expected_orientation

def test_select_invalid_key_returns_no_action():
    agent = HumanAgent((0, 0))
    agent._get_key = lambda: "a"
    state = Mock()
    expected_action = Action.NONE
    actual_action = agent.select_action(state)
    assert expected_action == actual_action
    assert agent.orientation == Action.RIGHT

def test_selecting_no_action_keeps_the_previous_orientation():
    agent = HumanAgent((0, 0))
    agent._get_key = lambda: ""
    agent.orientation = Action.UP
    state = Mock()
    expected_action = Action.NONE
    actual_action = agent.select_action(state)
    assert expected_action == actual_action
    assert agent.orientation == Action.UP

