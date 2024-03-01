from ghosts.ghost import Ghost
from tiles.tile import Tile

class ClydeGhost(Ghost):

    def __init__(self, location: tuple[int, int]):
        super().__init__(Tile.CLYDE, location)