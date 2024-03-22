import pytest
from unittest.mock import Mock, patch
import numpy as np
from agents.q_learning_agent import QLearningAgent
from models.q_table import QTable
from states.state import State
from actions.action import Action


# TODO: These unit tests needs to be implemented

@pytest.fixture()
def setup():
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3}
    q_table = QTable(np.zeros((20000, 5)))
    agent = QLearningAgent((1, 2), hyperparameters)
    return hyperparameters, q_table, agent

def test_init(setup):
    pass

@pytest.mark.parametrize("threshold, expected_action_id", [
    (0.2, 1),
    (0.3, 2),
    (0.4, 2)])
def test_select_action(threshold, expected_action_id, setup):
    pass

def test_update(setup):
    pass

def test_get_model(setup):
    pass
@pytest.mark.parametrize("model, expected_model", [
    (None, QTable(np.zeros((20000, 5)))),
    (QTable(np.ones((20000, 5))), QTable(np.ones((20000, 5))))])
def test_set_model(model, expected_model, setup):
    pass

@pytest.mark.parametrize("is_invincible, up, down, left, right, expected_state_id", [
    (True, 2, 3, 4, 5, 12345),
    (False, 2, 3, 4, 5, 2345),
    (False, 0, 1, 2, 3, 123),
    (True, 0, 1, 2, 3, 10123)])
def test_convert_state(is_invincible, up, down, left, right, expected_state_id, setup):
    pass



