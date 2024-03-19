from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile

class ClydeGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        super().__init__(location, house)
        self.tile = Tile.CLYDE
        self.scatter_target = (20, 0)
        self.wait_time = 40

    def _get_chase_target(self, agent_location: tuple[int, int], agent_orientation: int, ghost_locations: list[tuple[int, tuple[int, int]]]) -> tuple[int, int]:
        distance = self._calculate_distance(agent_location, self.location)
        if distance > 8:
            return agent_location
        else:
            return self.scatter_target
