import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile

class TestTiles():

    @classmethod
    def create(cls, array: [[int]]):
        tiles = []
        for row in array:
            tiles_row = []
            for tile_id in row:
                tile = Tile.get_enum_from_id(tile_id)
                tiles_row.append(tile)
            tiles.append(tiles_row)
        return Tiles(tiles)

    @classmethod
    def create_zeros(self, size: int):
        return self.create(np.zeros((size, size)))



