import numpy as np
from agents.agent import Agent
from models.feature_weights import FeatureWeights
from actions.action import Action
from states.state import State
from agents.feature_extraction import FeatureExtraction
from environments.transitions import get_action_transition
from tiles.tile import Tile

class ApproximateQLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.num_actions = 5
        self.num_features = hyperparameters["num_features"]
        self.feature_weights = np.zeros(self.num_features)

    def select_action(self, state: State) -> Action:
        if self._get_random_threshold() < self.epsilon:
            action_id = self._get_random_action_id()
        else:
            q_values = self._calculate_max_feature_vector(state)
            action_id = np.argmax(q_values)
        return Action(action_id)
    
    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        feature_vector = self._calculate_feature_vector(next_state)
        q_value = np.dot(self.feature_weights, feature_vector)
        max_q_value = max(self._calculate_max_feature_vector(next_state))
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
    
    def _calculate_feature_vector(self, state:State): 
        tiles = state.tiles
        feature_extraction = FeatureExtraction(tiles, state)
        closest_food_distance = feature_extraction.distance_closest_food()
        closest_ghost_distance = feature_extraction.distance_closest_ghost()
        num_active_ghosts_1 = feature_extraction.number_active_ghosts_1step()
        num_active_ghosts_2 = feature_extraction.number_active_ghosts_2step()
        num_scared_ghosts_1 = feature_extraction.number_scared_ghosts_1step()
        num_scared_ghosts_2 = feature_extraction.number_scared_ghosts_2step()

        feature_vector = np.array([closest_food_distance, closest_ghost_distance, num_active_ghosts_1, num_active_ghosts_2, 
                                   num_scared_ghosts_1, num_scared_ghosts_2])
        return feature_vector

    def _calculate_max_feature_vector(self, next_state:State):
        q_values = []
        actions = [Action.NONE, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        for action in actions: 
            next_state.agent_location = self.determine_new_location(next_state, action)
            feature_vector = self._calculate_feature_vector(next_state)
            q_values.append(np.dot(self.feature_weights, feature_vector))
        return np.array(q_values)
    
    def determine_new_location(self, next_state:State, action:Action): 
        current_location = next_state.agent_location
        transition = get_action_transition(action)
        new_row = current_location[0] + transition[0]
        new_col = current_location[1] + transition[1]
        new_location = (new_row, new_col)

        height = next_state.tiles.shape[0]
        width = next_state.tiles.shape[1]

        if self._can_teleport(new_location, height, width):
            new_location = self._teleport(new_location, height, width)
        
        if next_state.tiles[new_location] == Tile.WALL:
            new_location = current_location
        
        return new_location

    def _can_teleport(self, new_location: tuple[int, int], height, width) -> bool:
        if new_location[0] < 0 \
                or new_location[1] < 0 \
                or new_location[0] >= height \
                or new_location[1] >= width:
            return True
        return False

    def _teleport(self, new_location: tuple[int, int], height, width) -> tuple[int, int]:
        if new_location[0] < 0:
            new_location = (height - 1, new_location[1])
        if new_location[0] >= height:
            new_location = (0, new_location[1])
        if new_location[1] < 0:
            new_location = (new_location[0], width - 1)
        if new_location[1] >= self.width:
            new_location = (new_location[0], 0)
        return new_location
    

