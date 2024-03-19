import copy
import numpy as np
from environments.transitions import get_action_transition
from tiles.tiles import Tiles
from tiles.tile import Tile
from agents.agent import Agent
from ghosts.ghost import Ghost
from ghosts.ghost import Mode
from states.state import State
from actions.action import Action

INVINCIBLE_TIME = 25
GHOST_MODE_TIMES = [
    (Mode.SCATTER, 7),
    (Mode.CHASE, 20),
    (Mode.SCATTER, 7),
    (Mode.CHASE, 20),
    (Mode.SCATTER, 5),
    (Mode.CHASE, 20),
    (Mode.SCATTER, 5),
    (Mode.CHASE, 1000)]

class Environment:
    def __init__(self, tiles: Tiles, agent: Agent, ghosts: list[Ghost]):
        self._tiles = tiles
        self.height = tiles.shape[0]
        self.width = tiles.shape[1]
        self.agent = agent
        self.ghosts = ghosts
        self._ghost_spawn_locations = copy.deepcopy(ghosts)
        self._invincible_time = 0
        self._ghost_mode_map = GHOST_MODE_TIMES.copy()
        ghost_mode_row = self._ghost_mode_map.pop(0)
        self.ghost_mode = ghost_mode_row[0]
        self._ghost_mode_time = ghost_mode_row[1]
        self.game_time = 0
        self.reward = 0
        self.is_game_over = False
        self.is_winner = False

    def reset(self, environment_id):
        raise NotImplementedError("You must create a new environment using the Environment Factory")

    def get_state(self) -> State:
        tiles = self._tiles.to_integer_array()
        for ghost in self.ghosts:
            tiles[ghost.location] = ghost.tile.id
        tiles[self.agent.location] = Tile.PACMAN.id
        state = State(
            tiles,
            self.agent.location,
            self.agent.orientation.value,
            [(ghost.tile.id, ghost.location) for ghost in self.ghosts],
            self._is_invincible(),
            self.ghost_mode.value)
        return state

    def execute_action(self, action) -> tuple[State, int, bool]:
        self.reward = 0
        self.game_time += 1
        self._decrement_invincible_time()
        self._decrement_ghost_mode_time()
        self._move_agent(action)
        self._check_if_level_complete()
        self._check_if_ghosts_touching()
        self._move_ghosts()
        self._check_if_ghosts_touching()
        state = self.get_state()
        return state, self.reward, self.is_game_over

    def _is_invincible(self) -> bool:
        return self._invincible_time > 0

    def _decrement_invincible_time(self):
        if self._is_invincible():
            self._invincible_time -= 1

    def _decrement_ghost_mode_time(self):
        self._ghost_mode_time -= 1
        if self._ghost_mode_time <= 0:
            ghost_mode_row = self._ghost_mode_map.pop(0)
            self.ghost_mode = ghost_mode_row[0]
            self._ghost_mode_time = ghost_mode_row[1]

    def _is_valid_move(self, new_location: tuple[int, int]) -> bool:
        if self._tiles[new_location] == Tile.WALL:
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

    def _move_agent(self, action: Action):
        old_location = self.agent.location
        transition = get_action_transition(action)
        new_row = self.agent.location[0] + transition[0]
        new_col = self.agent.location[1] + transition[1]
        self.agent.location = (new_row, new_col)

        if self._can_teleport(self.agent.location):
            self.agent.location = self._teleport(self.agent.location)

        if self._tiles[self.agent.location] == Tile.WALL:
            self.agent.location = old_location
            self.reward = Tile.WALL.reward
        elif self._tiles[self.agent.location] == Tile.EMPTY:
            self.reward = Tile.EMPTY.reward
        elif self._tiles[self.agent.location] == Tile.DOT:
            self._tiles[self.agent.location] = Tile.EMPTY
            self.reward = Tile.DOT.reward
        elif self._tiles[self.agent.location] == Tile.POWER:
            self._tiles[self.agent.location] = Tile.EMPTY
            self.reward = Tile.POWER.reward
            self._invincible_time = INVINCIBLE_TIME

    def _check_if_level_complete(self):
        if not np.any(self._tiles == Tile.DOT):
            self.is_game_over = True
            self.is_winner = True

    def _check_if_ghosts_touching(self):
        for i, ghost in enumerate(self.ghosts):
            if ghost.location == self.agent.location:
                if self._is_invincible():
                    self.reward = Tile.STATIC.reward
                    self.ghosts[i] = self._ghost_spawn_locations[i]
                else:
                    self.is_game_over = True
                    self.is_winner = False

    def _move_ghosts(self):
        for i, ghost in enumerate(self.ghosts):
            action = ghost.select_action(self.get_state())
            transition = get_action_transition(action)
            new_row = ghost.location[0] + transition[0]
            new_col = ghost.location[1] + transition[1]
            new_location = (new_row, new_col)
            self.ghosts[i].location = new_location





