import numpy as np
from agents.agent import Agent
from models.model import Model
from models.q_table import QTable
from actions.action import Action
from states.state import State

class QLearningAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float]):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.num_actions = 5
        self.num_states = 20000
        self.q_table = np.zeros((self.num_states, self.num_actions))

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
        q_value = self.q_table[state_id, action_id]
        max_q_value = max(self.q_table[next_state_id, ])
        self.q_table[state_id, action_id] += self.alpha * (reward + self.gamma * max_q_value - q_value)

    def get_model(self) -> QTable:
        return QTable(self.q_table)

    def set_model(self, model: QTable) -> None:
        if model is None or model.table is None:
            self.q_table = np.zeros((self.num_states, self.num_actions))
        else:
            self.q_table = model.table

    def _get_random_threshold(self) -> float:
        return np.random.uniform()

    def _get_random_action_id(self) -> int:
        return np.random.choice(self.num_actions)
    
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