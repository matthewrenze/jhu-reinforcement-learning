import pytest
from unittest.mock import Mock, patch
import numpy as np
from agents.approximate_q_learning_agent import ApproximateQLearningAgent
from models.feature_weights import FeatureWeights
from states.state import State
from actions.action import Action

@pytest.fixture()
def setup():
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3, "features":[0,1]}
    feature_weights = FeatureWeights(np.zeros(8))
    agent = ApproximateQLearningAgent((1,2), hyperparameters)
    return hyperparameters, feature_weights, agent

def test_init(setup):
    hyperparameters, feature_weights, agent = setup
    assert agent.alpha == 0.1
    assert agent.gamma == 0.2
    assert agent.epsilon == 0.3
    assert len(agent.features) == 2
    assert agent.num_actions == 5
    assert agent.feature_weights.shape == (2,)

@pytest.mark.parametrize("threshold, expected_action_id", [
    (0.2, 1),
    (0.3, 2),
    (0.4, 2)])
def test_select_action(threshold, expected_action_id, setup):
    _, _, agent = setup
    state = Mock()
    agent._get_random_threshold = Mock(return_value=threshold)
    agent._get_random_action_id = Mock(return_value=1)
    agent._calculate_max_feature_vector = Mock(return_value = [0.0, 1.0, 2.0, 0.0, 0.0])
    action = agent.select_action(state)
    assert action.value == expected_action_id

def test_update(setup):
    _, _, agent = setup
    state = Mock()
    action = Action.UP
    reward = 1.1 
    next_state = Mock()
    agent._get_random_threshold = Mock(return_value=0.2)
    agent._get_random_action_id = Mock(return_value=1)
    agent._calculate_feature_vector = Mock(return_value = np.array([1, 4]))
    agent.feature_weights = np.array([0.5, 0.5])
    agent._calculate_max_feature_vector = Mock(return_value = [0.0, 1.0, 2.0, 0.0, 0.0])
    agent.update(state, action, reward, next_state)
    assert abs(agent.feature_weights[0]-0.4) < 0.001
    assert abs(agent.feature_weights[1]-0.1) < 0.001

def test_get_model(setup):
    _, feature_weights, agent = setup
    agent.feature_weights = feature_weights
    model = agent.get_model()
    assert np.array_equal(model.table, feature_weights)

@pytest.mark.parametrize("model, expected_model", [
    (None, FeatureWeights(np.zeros(2,))),
    (FeatureWeights(np.ones(2,)), FeatureWeights(np.ones((2, ))))])
def test_set_model(model, expected_model, setup):
    _, q_table, agent = setup
    agent.set_model(model)
    assert np.array_equal(agent.feature_weights, expected_model.table)





