import random
import numpy as np
from agents.agent import Agent
from models.feature_weights import FeatureWeights
from actions.action import Action
from states.state import State
from agents.feature_extraction import FeatureExtraction
from environments.legal_positions import find_legal_actions

class ApproximateQLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.features = hyperparameters["features"]
        self.num_actions = 5
        self.feature_weights = np.zeros(self.features)

    def select_action(self, state: State) -> Action:
        legal_actions = find_legal_actions(state.tiles, state.agent_location)
        if self._get_random_threshold() < self.epsilon:
            action_id = random.choice(legal_actions)
        else:
            q_values = self._calculate_max_feature_vector(state, legal_actions)
            try:
                action_id = np.random.choice(np.where(q_values == np.nanmax(q_values))[0])
            except:
                action_id = np.nanargmax(q_values)
        return Action(action_id)
    
    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        feature_vector = self._calculate_feature_vector(state, action)
        #print("Feature Vector: {}".format(feature_vector))
        feature_weights = self.feature_weights
        activation = True
        if len(feature_vector) < len(feature_weights):
            activation = False
            q_value = np.dot(feature_weights[:-1], feature_vector) 
        else: 
            q_value = np.dot(self.feature_weights, feature_vector)
        legal_actions = find_legal_actions(next_state.tiles, next_state.agent_location)
        max_q_value = np.nanmax(self._calculate_max_feature_vector(next_state, legal_actions))
        correction = reward + self.gamma * max_q_value - q_value       

        if not activation:
            tmp_weights = feature_weights[:-1] + (self.alpha*correction)*feature_vector
            self.feature_weights = np.append(tmp_weights, self.feature_weights[-1])
        else: 
            self.feature_weights = feature_weights + (self.alpha*correction)*feature_vector

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
    
    def _calculate_feature_vector(self, state:State, action:Action): 
        feature_extraction = FeatureExtraction(state, action)
        selected_features = self.features
        feature_dict = {
            0: 1,  # bias
            1: feature_extraction.distance_closest_food()/10,
            2: feature_extraction.distance_closest_ghost()/10,
            3: feature_extraction.distance_closest_powerpellet()/441,
            4: feature_extraction.number_active_ghosts_1step(),
            5: feature_extraction.number_active_ghosts_2step(),
            6: feature_extraction.number_scared_ghosts_1step(),
            7: feature_extraction.number_scared_ghosts_2step(),
            #8: feature_extraction.number_power_pellets_1step(),
            #9: feature_extraction.number_power_pellets_2steps(),
            #10: feature_extraction.number_food_1step(),
            #11: feature_extraction.number_food_2steps(),
            12: feature_extraction.food_focus(), 
            13: feature_extraction.safe_mode()}
         
        if 12 in selected_features and feature_dict[12] == 0: 
            feature_vector = np.array([feature_dict[i]/10 for i in selected_features[:-1]])
        else: 
            feature_vector = np.array([feature_dict[i]/10 for i in selected_features])

        return feature_vector

    def _calculate_max_feature_vector(self, state:State, legal_actions):
        feature_weights = self.feature_weights
        q_values = []
        all_actions = [Action.NONE, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        
        for i in range(len(all_actions)):
            if i in legal_actions:
                feature_vector = self._calculate_feature_vector(state, all_actions[i])
                if len(feature_vector) < len(feature_weights):
                    q_values.append(np.dot(feature_weights[:-1], feature_vector))
                else: 
                    q_values.append(np.dot(feature_weights, feature_vector))
            else: 
                q_values.append(np.nan)
        return np.array(q_values)

        

           
            
    


