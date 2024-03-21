import numpy as np
from agents.agent import Agent
from actions.action import Action
from states.state import State


class SarsaAgent(Agent):

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float], q_table: np.ndarray):
        super().__init__(location, hyperparameters)
        self.alpha = hyperparameters["alpha"]
        self.gamma = hyperparameters["gamma"]
        self.epsilon = hyperparameters["epsilon"]
        self.num_actions = 5
        self.num_states = 20000
        if q_table is None:
            self.q_table = np.zeros((self.num_states, self.num_actions))
        else:
            self.q_table = q_table

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

    def _get_random_threshold(self) -> float:
        return np.random.uniform()

    def _get_random_action_id(self) -> int:
        return np.random.choice(self.num_actions)

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
        next_action = self.select_action(next_state)
        next_action_id = next_action.value
        q_value = self.q_table[state_id, action_id]
        q_next_value = self.q_table[next_state_id, next_action_id]
        self.q_table[state_id, action_id] += self.alpha * (reward + self.gamma * q_next_value - q_value)
