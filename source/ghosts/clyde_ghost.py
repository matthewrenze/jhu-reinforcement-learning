from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile

class ClydeGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.CLYDE
        scatter_target = (20, 0)
        super().__init__(tile, location, scatter_target, house)