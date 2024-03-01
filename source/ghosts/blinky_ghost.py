from ghosts.ghost import Ghost
from tiles.tile import Tile

class BlinkyGhost(Ghost):

    def __init__(self, location: tuple[int, int]):
        super().__init__(Tile.BLINKY, location)

