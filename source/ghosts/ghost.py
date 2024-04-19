from enum import Enum
import numpy as np
from houses.house import House
from states.state import State
from tiles.tile import Tile
from actions.action import Action

class Mode(Enum):
    CHASE = 0
    SCATTER = 1
    FRIGHTENED = 2

class Ghost:

    def __init__(self, location: tuple[int, int], house: House):
        self.tile = None
        self.location = location
        self.orientation = Action.NONE
        self.spawn_location = location
        self.house_locations = house.house_locations
        self.house_exit_target = house.exit_target
        self.scatter_target = None
        self.mode = Mode.SCATTER
        self.wait_time = None

    def select_action(self, state: State) -> Action:

        tiles = state.tiles
        new_mode = Mode(state.ghost_mode)
        agent_location = state.agent_location
        agent_orientation = state.agent_orientation
        ghost_locations = state.ghost_locations

        if self.wait_time > 0:
            self.wait_time -= 1
            action = Action.NONE

        elif self._should_reverse(self.mode, new_mode):
            action = self._get_reverse(self.orientation)

        elif self._is_in_house(self.location):
            target = self.house_exit_target
            action = self._find_best_move(tiles, target)

        elif new_mode == Mode.CHASE:
            target = self._get_chase_target(agent_location, agent_orientation, ghost_locations)
            action = self._find_best_move(tiles, target)

        elif new_mode == Mode.FRIGHTENED:
            action = self._get_random_action(tiles)

        else:  # Scatter mode
            target = self.scatter_target
            action = self._find_best_move(tiles, target)

        self.mode = new_mode
        self.orientation = action

        return action

    def on_eaten(self):
        self.location = self.spawn_location
        self.orientation = Action.NONE
        self.wait_time = 5

    def _should_reverse(self, previous_mode: Mode, current_mode: Mode) -> bool:
        return previous_mode != current_mode

    def _get_reverse(self, orientation: Action) -> Action:
        if orientation == Action.UP:
            return Action.DOWN
        if orientation == Action.DOWN:
            return Action.UP
        if orientation == Action.LEFT:
            return Action.RIGHT
        if orientation == Action.RIGHT:
            return Action.LEFT
        return Action.NONE

    def _is_in_house(self, location: tuple[int, int]) -> bool:
        return location in self.house_locations

    def _get_chase_target(
            self,
            agent_location: tuple[int, int],
            agent_orientation: int,
            ghost_locations: list[tuple[int, tuple[int, int]]]) -> tuple[int, int]:
        raise NotImplementedError("You must implement the find_chase_target method")

    def _get_random_action(self, tiles) -> Action:
        while True:
            action_id = np.random.randint(1, 5)
            action = Action(action_id)
            new_location = self._get_new_location(self.location, action, tiles)
            if self._is_valid_move(tiles, new_location, action):
                return action

    def _find_best_move(self, tiles: np.ndarray, target: tuple[int, int]) -> Action:
        possible_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        best_action = Action.NONE
        min_distance = float('inf')

        for action in possible_actions:
            new_location = self._get_new_location(self.location, action, tiles)
            if self._is_valid_move(tiles, new_location, action):
                distance = self._calculate_distance(new_location, target)
                if distance < min_distance:
                    min_distance = distance
                    best_action = action

        return best_action

    def _get_new_location(self, location: tuple[int, int], action: Action, tiles: np.ndarray) -> tuple[int, int]:
        height = tiles.shape[0]
        width = tiles.shape[1]
        if action == Action.UP:
            return (location[0] - 1) % height, location[1]
        if action == Action.DOWN:
            return (location[0] + 1) % height, location[1]
        if action == Action.LEFT:
            return location[0], (location[1] - 1) % width
        if action == Action.RIGHT:
            return location[0], (location[1] + 1) % width

    def _is_valid_move(self, tiles: np.ndarray, location: tuple[int, int], action: Action) -> bool:
        if self._is_reverse(action):
            return False
        if self._is_wall(tiles, location):
            return False
        return True

    def _is_reverse(self, action: Action):
        if action == Action.NONE:
            return False
        reverse = self._get_reverse(action)
        return self.orientation == reverse

    def _is_wall(self, tiles: np.ndarray, location: tuple[int, int]) -> bool:
        return tiles[location[0], location[1]] == Tile.WALL.id

    def _calculate_distance(self, from_location: tuple[int, int], to_location: tuple[int, int]) -> float:
        return np.sqrt((to_location[0] - from_location[0]) ** 2 + (to_location[1] - from_location[1]) ** 2)