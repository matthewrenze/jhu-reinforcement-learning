import numpy as np
from tiles.tile_factory import TileFactory
from tiles.test_tiles import TestTiles

def test_create_from_str():
    factory = TileFactory()
    map = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    expected_tiles = TestTiles.create([[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 3, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]])
    tiles = factory.create(map)
    assert np.array_equal(tiles, expected_tiles)