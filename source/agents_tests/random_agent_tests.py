import pytest
from unittest.mock import Mock
from agents.random_agent import RandomAgent
from actions.action import Action

@pytest.mark.parametrize("action_id, expected_action, expected_orientation", [
    (0, Action.NONE, Action.RIGHT),
    (1, Action.UP, Action.UP),
    (2, Action.DOWN, Action.DOWN),
    (3, Action.LEFT, Action.LEFT),
    (4, Action.RIGHT, Action.RIGHT)])
def test_select_action(action_id, expected_action, expected_orientation):
    agent = RandomAgent((0, 0))
    agent._get_random_action_id = lambda: action_id
    state = Mock()
    action = agent.select_action(state)
    assert action == expected_action
    assert agent.orientation == expected_orientation
