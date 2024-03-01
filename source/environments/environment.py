import copy
import numpy as np
from environments.transitions import get_action_transition
from tiles.tile import Tile
from agents.agent import Agent
from ghosts.ghost import Ghost


INVINCIBLE_TIME = 25

class Environment:
    def __init__(self, state: np.ndarray, agent: Agent, ghosts: list[Ghost]):
        self._state = state
        self.height = state.shape[0]
        self.width = state.shape[1]
        self.agent = agent
        self.ghosts = ghosts
        self._ghost_respawns = copy.deepcopy(ghosts)
        self.invincible_time = 0
        self.reward = 0
        self.is_game_over = False
        self.is_winner = False

        # TODO: Move this into the environment factory
        self._state[agent.location] = Tile.EMPTY.id
        for ghost in ghosts:
            self._state[ghost.location] = Tile.EMPTY.id

    def reset(self, environment_id):
        raise NotImplementedError("You must create a new environment using the Environment Factory")

    def get_state(self) -> np.ndarray:
        state = np.copy(self._state)
        for ghost in self.ghosts:
            state[ghost.location] = ghost.tile.id
        state[self.agent.location] = Tile.PACMAN.id
        return state

    def _is_invincible(self) -> bool:
        return self.invincible_time > 0

    def _decrement_invincible_time(self):
        if self._is_invincible():
            self.invincible_time -= 1

    def _is_valid_move(self, new_location: tuple[int, int]) -> bool:
        if self._state[new_location] == Tile.WALL.id:
            return False
        return True

    def _can_teleport(self, new_location: tuple[int, int]) -> bool:
        if new_location[0] < 0 \
                or new_location[1] < 0 \
                or new_location[0] >= self.height \
                or new_location[1] >= self.width:
            return True
        return False

    def _teleport(self, new_location: tuple[int, int]) -> tuple[int, int]:
        if new_location[0] < 0:
            new_location = (self.height - 1, new_location[1])
        if new_location[0] >= self.height:
            new_location = (0, new_location[1])
        if new_location[1] < 0:
            new_location = (new_location[0], self.width - 1)
        if new_location[1] >= self.width:
            new_location = (new_location[0], 0)
        return new_location

    def _move_agent(self, action: int):
        old_location = self.agent.location
        transition = get_action_transition(action)
        new_row = self.agent.location[0] + transition[0]
        new_col = self.agent.location[1] + transition[1]
        self.agent.location = (new_row, new_col)

        if self._can_teleport(self.agent.location):
            self.agent.location = self._teleport(self.agent.location)

        if self._state[self.agent.location] == Tile.WALL.id:
            self.agent.location = old_location
            self.reward = Tile.WALL.reward
        elif self._state[self.agent.location] == Tile.EMPTY.id:
            self.reward = Tile.EMPTY.reward
        elif self._state[self.agent.location] == Tile.DOT.id:
            self._state[self.agent.location] = Tile.EMPTY.id
            self.reward = Tile.DOT.reward
        elif self._state[self.agent.location] == Tile.POWER.id:
            self._state[self.agent.location] = Tile.EMPTY.id
            self.reward = Tile.POWER.reward
            self.invincible_time = INVINCIBLE_TIME

    def _check_if_level_complete(self):
        if not np.any(self._state == Tile.DOT.id):
            self.is_game_over = True
            self.is_winner = True

    def _check_if_ghosts_touching(self):
        for i, ghost in enumerate(self.ghosts):
            if ghost.location == self.agent.location:
                if self._is_invincible():
                    self.reward = Tile.STATIC.reward
                    self.ghosts[i] = self._ghost_respawns[i]
                else:
                    self.is_game_over = True
                    self.is_winner = False

    def _move_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            action = ghost.select_action(self.get_state(), self.agent.location, True)
            transition = get_action_transition(action)
            new_row = ghost.location[0] + transition[0]
            new_col = ghost.location[1] + transition[1]
            new_location = (new_row, new_col)
            if self._is_valid_move(new_location):
                self.ghosts[i].location = new_location

    def execute_action(self, action) -> tuple[np.ndarray, int, bool]:
        self.reward = 0
        self._decrement_invincible_time()
        self._move_agent(action)
        self._check_if_level_complete()
        self._check_if_ghosts_touching()
        self._move_ghosts()
        self._check_if_ghosts_touching()
        state = self.get_state()
        return state, self.reward, self.is_game_over





