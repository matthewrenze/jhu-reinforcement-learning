import numpy as np
from houses.house import House
from ghosts.ghost import Ghost
from tiles.tiles import Tiles
from tiles.tile import Tile
from ghosts.static_ghost import StaticGhost
from ghosts.blinky_ghost import BlinkyGhost
from ghosts.pinky_ghost import PinkyGhost
from ghosts.inky_ghost import InkyGhost
from ghosts.clyde_ghost import ClydeGhost

class GhostFactory:

    def _get_ghosts(self, tiles: Tiles) -> list[tuple[Tile, int, int]]:
        ghosts = []
        for ghost in [Tile.STATIC, Tile.BLINKY, Tile.PINKY, Tile.INKY, Tile.CLYDE]:
            locations = np.where(tiles == ghost)
            locations = list(zip(locations[0], locations[1]))
            for location in locations:
                ghosts.append((ghost, location[0], location[1]))
        return ghosts

    def _create_ghost(self, tile: Tile, location: tuple[int, int], house: House) -> Ghost:
        if tile == Tile.STATIC:
            return StaticGhost(location, house)
        elif tile == Tile.BLINKY:
            return BlinkyGhost(location, house)
        elif tile == Tile.PINKY:
            return PinkyGhost(location, house)
        elif tile == Tile.INKY:
            return InkyGhost(location, house)
        elif tile == Tile.CLYDE:
            return ClydeGhost(location, house)

    def create(self, tiles: Tiles, house: House) -> list[Ghost]:
        ghosts = []
        ghost_tuple = self._get_ghosts(tiles)
        for ghost_tuples in ghost_tuple:
            tile = ghost_tuples[0]
            location = (ghost_tuples[1], ghost_tuples[2])
            ghost = self._create_ghost(tile, location, house)
            ghosts.append(ghost)
        return ghosts

