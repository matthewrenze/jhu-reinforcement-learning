import numpy as np
import pytest
from tiles.tiles import Tiles
from tiles.test_tiles import TestTiles
from tiles.tile import Tile

@pytest.fixture
def setup():
    tiles = np.array([
        [Tile.WALL, Tile.WALL, Tile.WALL, Tile.WALL, Tile.WALL],
        [Tile.WALL, Tile.DOT, Tile.STATIC, Tile.DOT, Tile.WALL],
        [Tile.WALL, Tile.DOT, Tile.WALL, Tile.DOT, Tile.WALL],
        [Tile.WALL, Tile.DOT, Tile.PACMAN, Tile.DOT, Tile.WALL],
        [Tile.WALL, Tile.WALL, Tile.WALL, Tile.WALL, Tile.WALL]])
    int_array = [[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 3, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]]
    return tiles, int_array

def test_new(setup):
    tiles, int_array = setup
    actual_tiles = Tiles(tiles)
    expected_tiles = TestTiles.create(int_array)
    assert np.array_equal(actual_tiles, expected_tiles)

def test_array_finalize(setup):
    tiles, int_array = setup
    actual_tiles = Tiles(tiles)
    assert actual_tiles.extra_info is None

def test_to_integer_array(setup):
    tiles, int_array = setup
    actual_tiles = Tiles(tiles)
    expected_array = np.array(int_array)
    assert np.array_equal(actual_tiles.to_integer_array(), expected_array)

