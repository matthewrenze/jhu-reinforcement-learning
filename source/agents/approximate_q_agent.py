import numpy as np
from agents.agent import Agent
from models.feature_weights import FeatureWeights
from actions.action import Action
from states.state import State

class ApproximateQLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        #self.num_states = 625 
        self.num_actions = 5
        self.num_features = 8
        self.feature_weights = np.zeros(self.num_features)

    def select_action(self, state: State) -> Action:
        state_id = self._convert_state(state)
        if self._get_random_threshold() < self.epsilon:
            action_id = self._get_random_action_id()
        else:
            action_id = np.argmax(self.q_table[state_id, :])
        return Action(action_id)
    
    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        state_id = self._convert_state(state)
        action_id = action.value
        next_state_id = self._convert_state(next_state)
        feature_vector = self._calculate_feature_vector(state_id, action_id)
        q_value = np.dot(self.feature_weights, feature_vector)
        max_q_value = self._calculate_max_feature_vector(next_state_id)
        correction = reward + self.gamma * max_q_value - q_value
        self.feature_weights = self.feature_weights + (self.alpha*correction)*feature_vector

    def get_model(self) -> FeatureWeights:
        return FeatureWeights(self.feature_weights)

    def set_model(self, model: FeatureWeights) -> None:
        if model is None or model.table is None:
            self.feature_weights = np.zeros(self.num_features)
        else:
            self.feature_weights = model.table

    def _get_random_threshold(self) -> float:
        return np.random.uniform()

    def _get_random_action_id(self) -> int:
        return np.random.choice(self.num_actions)
    
    def _calculate_feature_vector(self, state_id, action_id): 
        pass

    def _calculate_max_feature_vector(self, state_id): 
        pass
    
    # TODO: Refactor this into an abstract tabular_agent superclass or state_converter class
    # TODO: To be shared by both SarsaAgent and QLearningAgent
    def _convert_state(self, state: State) -> int:
        agent_location = state.agent_location
        is_invincible = int(state.is_invincible)
        tiles = state.tiles

        height = tiles.shape[0]
        width = tiles.shape[1]

        up = tiles[(agent_location[0] - 1) % height, agent_location[1]]
        down = tiles[(agent_location[0] + 1) % height, agent_location[1]]
        left = tiles[agent_location[0], (agent_location[1] - 1) % width]
        right = tiles[agent_location[0], (agent_location[1] + 1) % width]

        state_str = f"{is_invincible}{up}{down}{left}{right}"
        state_str = state_str.lstrip("0")
        if len(state_str) == 0:
            state_str = "0"
        state_id = int(state_str)

        return state_id
