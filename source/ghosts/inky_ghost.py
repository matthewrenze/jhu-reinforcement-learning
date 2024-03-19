from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile
from actions.action import Action

class InkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        super().__init__(location, house)
        self.tile = Tile.INKY
        self.scatter_target = (20, 20)
        self.wait_time = 20

    def _get_chase_target(
            self,
            agent_location: tuple[int, int],
            agent_orientation: int,
            ghost_locations: [list[tuple[int, tuple[int, int]]]]) -> tuple[int, int]:
        blinky = next((ghost for ghost in ghost_locations if ghost[0] == Tile.BLINKY.id), None)
        if blinky is None:
            return agent_location
        blinky_location = blinky[1]
        two_tiles_ahead = self.get_two_tiles_ahead(agent_location, agent_orientation)
        scaled_vector = self.calculate_scaled_vector(blinky_location, two_tiles_ahead)
        target_location = (blinky_location[0] + scaled_vector[0], blinky_location[1] + scaled_vector[1])
        return target_location


    def get_two_tiles_ahead(self, agent_location: tuple[int, int], agent_orientation: int) -> tuple[int, int]:
        if agent_orientation == Action.UP.value:
            return (agent_location[0] - 2, agent_location[1])
        elif agent_orientation == Action.DOWN.value:
            return (agent_location[0] + 2, agent_location[1])
        elif agent_orientation == Action.LEFT.value:
            return (agent_location[0], agent_location[1] - 2)
        elif agent_orientation == Action.RIGHT.value:
            return (agent_location[0], agent_location[1] + 2)
        else:
            return agent_location

    def calculate_scaled_vector(self, blinky_location: tuple[int, int], two_tiles_ahead: tuple[int, int]) -> tuple[int, int]:
        vector = (two_tiles_ahead[0] - blinky_location[0], two_tiles_ahead[1] - blinky_location[1])
        scaled_vector = (vector[0] * 2, vector[1] * 2)
        return scaled_vector