import copy
import numpy as np
from agents.agent import Agent
from models.feature_weights import FeatureWeights
from actions.action import Action
from states.state import State
from agents.feature_extraction import FeatureExtraction

class ApproximateQLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.features = hyperparameters["features"]
        self.num_actions = 5
        self.feature_weights = np.zeros(len(self.features))

    def select_action(self, state: State) -> Action:
        if self._get_random_threshold() < self.epsilon:
            action_id = self._get_random_action_id()
        else:
            q_values = self._calculate_max_feature_vector(state)
            try:
                action_id = np.random.choice(np.where(q_values == q_values.max())[0])
            except:
                action_id = np.argmax(q_values)
        return Action(action_id)
    
    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        feature_vector = self._calculate_feature_vector(state)
        q_value = np.dot(self.feature_weights, feature_vector)
        max_q_value = max(self._calculate_max_feature_vector(next_state))
        correction = reward + self.gamma * max_q_value - q_value        
        self.feature_weights = self.feature_weights + (self.alpha*correction)*feature_vector

    def get_model(self) -> FeatureWeights:
        return FeatureWeights(self.feature_weights)

    def set_model(self, model: FeatureWeights) -> None:
        if model is None or model.table is None:
            self.feature_weights = np.zeros(len(self.features))
        else:
            self.feature_weights = model.table

    def _get_random_threshold(self) -> float:
        return np.random.uniform()

    def _get_random_action_id(self) -> int:
        return np.random.choice(self.num_actions)
    
    def _calculate_feature_vector(self, state:State): 
        feature_extraction = FeatureExtraction(state)
        selected_features = self.features
        feature_dict = {
            0: 1,  # bias
            1: feature_extraction.distance_closest_food(),
            2: feature_extraction.distance_closest_ghost(),
            3: feature_extraction.distance_closest_powerpellet(),
            4: feature_extraction.number_active_ghosts_1step(),
            5: feature_extraction.number_active_ghosts_2step(),
            6: feature_extraction.number_scared_ghosts_1step(),
            7: feature_extraction.number_scared_ghosts_2step(),
            8: feature_extraction.number_power_pellets_1step(),
            9: feature_extraction.number_power_pellets_2steps(),
            10: feature_extraction.number_food_1step(),
            11: feature_extraction.number_food_2steps()}
        
        feature_vector = np.array([feature_dict[i] for i in selected_features])
        return feature_vector

    def _calculate_max_feature_vector(self, next_state:State):
        feature_extraction = FeatureExtraction(next_state)
        state_copy = copy.deepcopy(next_state)
        q_values = []
        possible_locations = feature_extraction.find_legal_positions(next_state.agent_location)[1]
        
        for i in range(5):
            state_copy.agent_location = possible_locations[i]
            feature_vector = self._calculate_feature_vector(state_copy)
            q_values.append(np.dot(self.feature_weights, feature_vector))

        return np.array(q_values)
    


