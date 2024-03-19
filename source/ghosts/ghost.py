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
    EATEN = 3

class Ghost:

    def __init__(self, tile: Tile, location: tuple[int, int], scatter_target: tuple[int, int], house: House):
        self.tile = tile
        self.location = location
        self.orientation = Action.NONE
        self.spawn_location = location
        self.house_locations = house.house_locations
        self.house_exit_target = house.exit_target
        self.scatter_target = scatter_target
        self.mode = Mode.SCATTER

    def select_action(self, state: State) -> Action:

        tiles = state.tiles
        mode = self.get_mode()

        if self.should_reverse(self.mode, mode):
            action = self.get_reverse(self.orientation)

        elif self.is_in_house(self.location):
            target = self.house_exit_target
            action = self.find_best_move(tiles, target)

        # elif not self.is_intersection(tiles, self.location):
        #     action = self.orientation

        elif self.mode == Mode.CHASE:
            target = state.agent_location  # NOTE: Need to implement custom get_target methods for each ghost
            action = self.find_best_move(tiles, target)
        else:
            target = self.scatter_target
            action = self.find_best_move(tiles, target)

        self.mode = mode
        self.orientation = action

        return action

    def get_mode(self) -> Mode:
        return self.mode

    def should_reverse(self, previous_mode: Mode, current_mode: Mode) -> bool:
        return previous_mode != current_mode

    def get_reverse(self, orientation: Action) -> Action:
        if orientation == Action.UP:
            return Action.DOWN
        if orientation == Action.DOWN:
            return Action.UP
        if orientation == Action.LEFT:
            return Action.RIGHT
        if orientation == Action.RIGHT:
            return Action.LEFT
        return Action.NONE

    def is_in_house(self, location: tuple[int, int]) -> bool:
        return location in [l for l in self.house_locations]

    # def is_intersection(self, tiles: np.ndarray, location: tuple[int, int]) -> bool:
    #     turns = (not self.is_wall(tiles, (location[0] - 1, location[1]))) \
    #         + (not self.is_wall(tiles, (location[0] + 1, location[1]))) \
    #         + (not self.is_wall(tiles, (location[0], location[1] - 1))) \
    #         + (not self.is_wall(tiles, (location[0], location[1] + 1)))
    #     return turns > 2

    def find_best_move(self, tiles: np.ndarray, target: tuple[int, int]) -> Action:
        possible_actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        best_action = Action.NONE
        min_distance = float('inf')

        for action in possible_actions:
            new_location = self.get_new_location(self.location, action)
            if self.is_valid_move(tiles, new_location, action):
                distance = self.calculate_distance(new_location, target)
                if distance < min_distance:
                    min_distance = distance
                    best_action = action

        return best_action

    def get_new_location(self, location: tuple[int, int], action: Action) -> tuple[int, int]:
        if action == Action.UP:
            return (location[0] - 1, location[1])
        elif action == Action.DOWN:
            return (location[0] + 1, location[1])
        elif action == Action.LEFT:
            return (location[0], location[1] - 1)
        elif action == Action.RIGHT:
            return (location[0], location[1] + 1)
        return location

    def is_valid_move(self, tiles: np.ndarray, location: tuple[int, int], action: Action) -> bool:

        if self.is_reverse(action):
            return False

        if self.is_wall(tiles, location):
            return False

        return True

    def is_reverse(self, action: Action):

        if action == Action.NONE:
            return False

        reverse = self.get_reverse(action)
        return self.orientation == reverse

    def is_wall(self, tiles: np.ndarray, location: tuple[int, int]) -> bool:
        return tiles[location[0], location[1]] == Tile.WALL.id

    def calculate_distance(self, from_location: tuple[int, int], to_location: tuple[int, int]) -> float:
        return np.sqrt((to_location[0] - from_location[0]) ** 2 + (to_location[1] - from_location[1]) ** 2)