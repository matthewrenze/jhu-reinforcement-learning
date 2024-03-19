from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile

class PinkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.PINKY
        scatter_target = (0, 0)
        super().__init__(tile, location, scatter_target, house)
