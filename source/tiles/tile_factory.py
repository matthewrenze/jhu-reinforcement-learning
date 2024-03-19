import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile

class TileFactory:

    def create(self, map: str):
        tiles = []
        rows = map.split('\n')
        for row in rows:
            tiles_row = []
            row = row.replace('|', '   ')
            for index, char in enumerate(row):
                if index % 3 != 0:
                    continue
                tile_id = Tile.get_enum_from_symbol(char)
                tiles_row.append(tile_id)
            tiles.append(tiles_row)
        tiles = Tiles(tiles)
        return tiles