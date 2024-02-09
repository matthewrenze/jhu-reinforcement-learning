from typing import Tuple
import numpy as np
from environments.transitions import get_action_transition
from environments.states import Tile

class Environment:
    def __init__(self, state: np.ndarray, agent_location: Tuple[int, int], ghost_locations: list[Tuple[int, int]]):
        self._state = state
        self.height = state.shape[0]
        self.width = state.shape[1]
        self._agent_location = agent_location
        self._state[agent_location] = Tile.EMPTY.id
        self._ghost_locations = ghost_locations
        for ghost_location in ghost_locations:
            self._state[ghost_location] = Tile.EMPTY.id

    def reset(self, environment_id):
        raise NotImplementedError("You must create a new environment using the Environment Factory")

    def get_state(self) -> np.ndarray:
        state = np.copy(self._state)
        state[self._agent_location] = Tile.PACMAN.id
        for ghost_location in self._ghost_locations:
            state[ghost_location] = Tile.GHOST.id
        return state

    def _is_valid_move(self, new_location: Tuple[int, int]) -> bool:
        if (new_location[0] < 0
                or new_location[0] >= self.height
                or new_location[1] < 0
                or new_location[1] >= self.width):
            return False
        if self._state[new_location] == Tile.WALL.id:
            return False
        return True

    def _has_dots(self) -> bool:
        return np.any(self._state == Tile.DOT.id)

    def execute_action(self, action) -> Tuple[np.ndarray, int, bool]:

        # Move the agent
        transition = get_action_transition(action)
        new_row = self._agent_location[0] + transition[0]
        new_col = self._agent_location[1] + transition[1]
        new_location = (new_row, new_col)
        reward = 0
        is_game_over = False
        if self._is_valid_move(new_location):
            self._agent_location = new_location
        if self._state[self._agent_location] == Tile.DOT.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.DOT.reward
        if self._state[self._agent_location] == Tile.POWER.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.POWER.reward
        if self._state[self._agent_location] == Tile.BONUS.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.BONUS.reward
        if self._state[self._agent_location] == Tile.GHOST.id:
            is_game_over = True
        if not self._has_dots():
            is_game_over = True

        # Check if ghosts touch agent (pre-check)
        for ghost_location in self._ghost_locations:
            if ghost_location == self._agent_location:
                is_game_over = True

        # Move the ghosts and check if they touch the agent (post-check)
        for index, ghost_location in enumerate(self._ghost_locations):
            transition = get_action_transition(np.random.randint(1, 5))
            new_row = ghost_location[0] + transition[0]
            new_col = ghost_location[1] + transition[1]
            new_location = (new_row, new_col)
            if self._is_valid_move(new_location):
                self._ghost_locations[index] = new_location

        # Check if ghosts touch agent (post-check)
        for ghost_location in self._ghost_locations:
            if ghost_location == self._agent_location:
                is_game_over = True

        state = self.get_state()
        return state, reward, is_game_over





