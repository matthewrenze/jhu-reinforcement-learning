from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile

class ClydeGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.CLYDE
        scatter_target = (20, 0)
        super().__init__(tile, location, scatter_target, house)

    def get_chase_target(self, agent_location: tuple[int, int], agent_orientation: int, ghost_locations: list[tuple[int, tuple[int, int]]]) -> tuple[int, int]:
        distance = self.calculate_distance(agent_location, self.location)
        if distance > 8:
            return agent_location
        else:
            return self.scatter_target
