import pytest
from unittest.mock import Mock, patch
import numpy as np
from agents.sarsa_agent import SarsaAgent
from states.state import State
from actions.action import Action

@pytest.fixture()
def setup():
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3}
    q_table = np.zeros((20000, 5))
    agent = SarsaAgent((1, 2), hyperparameters)
    return hyperparameters, q_table, agent

def test_init(setup):
    hyperparameters, q_table, agent = setup
    assert agent.alpha == 0.1
    assert agent.gamma == 0.2
    assert agent.epsilon == 0.3
    assert agent.num_actions == 5
    assert agent.num_states == 20000
    assert agent.q_table.shape == (20000, 5)

@pytest.mark.parametrize("threshold, expected_action_id", [
    (0.2, 1),
    (0.3, 2),
    (0.4, 2)])
def test_select_action(threshold, expected_action_id, setup):
    _, _, agent = setup
    state = Mock()
    agent._convert_state = Mock(return_value=0)
    agent._get_random_threshold = Mock(return_value=threshold)
    agent._get_random_action_id = Mock(return_value=1)
    agent.q_table = np.array([[0.0, 1.0, 2.0, 0.0, 0.0]])
    action = agent.select_action(state)
    assert action.value == expected_action_id

def test_update(setup):
    _, _, agent = setup
    state = Mock()
    action = Action.UP
    reward = 1.0
    next_state = Mock()
    agent._convert_state = Mock(return_value=0)
    agent._get_random_threshold = Mock(return_value=0.2)
    agent._get_random_action_id = Mock(return_value=1)
    agent.q_table = np.array([[0.0, 1.0, 2.0, 0.0, 0.0]])
    agent.update(state, action, reward, next_state)
    assert agent.q_table[0, 1] == 1.02

@patch("os.path.exists")
@patch("numpy.loadtxt")
def test_load(mock_loadtxt, mock_exists, setup):
    _, _, agent = setup
    mock_exists.return_value = True
    agent.load()
    mock_loadtxt.assert_called_with("../models/sarsa.csv", delimiter=",")

@patch("numpy.savetxt")
def test_save(mock_savetxt, setup):
    _, q_table, agent = setup
    agent.save()
    # NOTE: The following complexity is necessary to test np.ndarray equality
    assert mock_savetxt.called
    call_args, call_kwargs = mock_savetxt.call_args
    assert call_args[0] == "../models/sarsa.csv"
    assert call_kwargs['delimiter'] == ","
    np.testing.assert_array_equal(call_args[1], q_table)

@pytest.mark.parametrize("is_invincible, up, down, left, right, expected_state_id", [
    (True, 2, 3, 4, 5, 12345),
    (False, 2, 3, 4, 5, 2345),
    (False, 0, 1, 2, 3, 123),
    (True, 0, 1, 2, 3, 10123)])
def test_convert_state(is_invincible, up, down, left, right, expected_state_id, setup):
    tiles = np.array([[0, up, 0], [left, 0, right], [0, down, 0]])
    state = State(tiles, (1, 1), 0, [], is_invincible, 0)
    _, _, agent = setup
    state_id = agent._convert_state(state)
    assert state_id == expected_state_id



