from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile


class BlinkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.BLINKY
        scatter_target = (0, 16)
        super().__init__(tile, location, scatter_target, house)

    def get_chase_target(
            self,
            agent_location: tuple[int, int],
            agent_orientation: int,
            ghost_locations: [list[tuple[int, tuple[int, int]]]]) -> tuple[int, int]:
        return agent_location

