from typing import Tuple
import copy
import numpy as np
from environments.transitions import get_action_transition
from environments.states import Tile

INVINCIBLE_TIME = 25

class Environment:
    def __init__(self, state: np.ndarray, agent_location: Tuple[int, int], ghost_locations: list[Tuple[int, int]]):
        self._state = state
        self.height = state.shape[0]
        self.width = state.shape[1]
        self._agent_location = agent_location
        self._state[agent_location] = Tile.EMPTY.id
        self._ghost_locations = ghost_locations
        self._original_ghost_locations = copy.deepcopy(ghost_locations)
        self.invincible_time = 0
        self._is_game_over = False
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
        if self._state[new_location] == Tile.WALL.id:
            return False
        return True

    def _can_teleport(self, new_location: Tuple[int, int]) -> bool:
        if new_location[0] < 0 \
                or new_location[1] < 0 \
                or new_location[0] >= self.height \
                or new_location[1] >= self.width:
            return True
        return False

    def _teleport(self, new_location: Tuple[int, int]) -> bool:
        if new_location[0] < 0:
            new_location = (self.height - 1, new_location[1])
        if new_location[0] >= self.height:
            new_location = (0, new_location[1])
        if new_location[1] < 0:
            new_location = (new_location[0], self.width - 1)
        if new_location[1] >= self.width:
            new_location = (new_location[0], 0)
        return new_location


    def _has_dots(self) -> bool:
        return np.any(self._state == Tile.DOT.id)

    def _is_invincible(self) -> bool:
        return self.invincible_time > 0

    def _get_random_action(self):
        return np.random.randint(1, 5)

    def execute_action(self, action) -> Tuple[np.ndarray, int, bool]:

        if self._is_invincible():
            self.invincible_time -= 1

        # Calculate the new position of the agent
        old_location = self._agent_location
        transition = get_action_transition(action)
        new_row = self._agent_location[0] + transition[0]
        new_col = self._agent_location[1] + transition[1]
        self._agent_location = (new_row, new_col)
        reward = 0

        if self._can_teleport(self._agent_location):
            self._agent_location = self._teleport(self._agent_location)

        if self._state[self._agent_location] == Tile.WALL.id:
            self._agent_location = old_location
            reward = Tile.WALL.reward
        elif self._state[self._agent_location] == Tile.EMPTY.id:
            reward = Tile.EMPTY.reward
        elif self._state[self._agent_location] == Tile.DOT.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.DOT.reward
        elif self._state[self._agent_location] == Tile.POWER.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.POWER.reward
            self.invincible_time = INVINCIBLE_TIME
        elif self._state[self._agent_location] == Tile.BONUS.id:
            self._state[self._agent_location] = Tile.EMPTY.id
            reward = Tile.BONUS.reward

        if not self._has_dots():
            self._is_game_over = True

        # Check if pacman is touching a ghost  (pre-check)
        # TODO: Refactor this into a single method
        for i, ghost_location in enumerate(self._ghost_locations):
            if ghost_location == self._agent_location:
                if self._is_invincible():
                    reward = Tile.GHOST.reward
                    self._ghost_locations[i] = self._original_ghost_locations[i]
                else:
                    self._is_game_over = True

        # Move the ghosts
        for index, ghost_location in enumerate(self._ghost_locations):
            action = self._get_random_action()
            transition = get_action_transition(action)
            new_row = ghost_location[0] + transition[0]
            new_col = ghost_location[1] + transition[1]
            new_location = (new_row, new_col)
            if self._is_valid_move(new_location):
                self._ghost_locations[index] = new_location

        # Check if a ghosts is touching pacman (post-check)
        # TODO: Refactor this into a single method
        for i, ghost_location in enumerate(self._ghost_locations):
            if ghost_location == self._agent_location:
                if self._is_invincible():
                    reward = Tile.GHOST.reward
                    self._ghost_locations[i] = self._original_ghost_locations[i]
                else:
                    self._is_game_over = True

        state = self.get_state()
        return state, reward, self._is_game_over





