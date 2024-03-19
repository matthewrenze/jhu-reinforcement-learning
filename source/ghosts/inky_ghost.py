from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile

class InkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.INKY
        scatter_target = (20, 20)
        super().__init__(tile, location, scatter_target, house)
