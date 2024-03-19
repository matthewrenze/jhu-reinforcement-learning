import pytest
from unittest.mock import Mock
from agents.human_agent import HumanAgent
from actions.action import Action


@pytest.mark.parametrize("key, expected_action", [
    ("i", Action.UP),
    ("l", Action.RIGHT),
    ("k", Action.DOWN),
    ("j", Action.LEFT),
    ("", Action.NONE)
])
def test_select_action(key, expected_action):
    agent = HumanAgent((0, 0))
    agent._get_key = lambda: key
    state = Mock()
    actual_action = agent.select_action(state)
    assert expected_action == actual_action
    assert agent.orientation == expected_action

def test_select_invalid_key_returns_no_action():
    agent = HumanAgent((0, 0))
    agent._get_key = lambda: "a"
    state = Mock()
    expected_action = Action.NONE
    actual_action = agent.select_action(state)
    assert expected_action == actual_action
    assert agent.orientation == expected_action

