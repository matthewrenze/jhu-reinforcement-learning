from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile


class BlinkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        super().__init__(location, house)
        self.tile = Tile.BLINKY
        self.scatter_target = (0, 16)
        self.wait_time = 0

    def get_chase_target(
            self,
            agent_location: tuple[int, int],
            agent_orientation: int,
            ghost_locations: [list[tuple[int, tuple[int, int]]]]) -> tuple[int, int]:
        return agent_location

