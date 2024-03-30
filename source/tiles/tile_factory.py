import numpy as np
from tiles.tiles import Tiles
from tiles.tile import Tile

class TileFactory:

    def create(self, level: int) -> Tiles:
        map = self._load(level)
        tiles = self._convert(map)
        return tiles

    def _load(self, environment_id: int) -> str:
        file_name = f"level-{environment_id}.txt"
        file_path = f"../data/levels/{file_name}"
        with open(file_path, 'r') as file:
            environment = file.read()
        return environment

    def _convert(self, map: str) -> Tiles:
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