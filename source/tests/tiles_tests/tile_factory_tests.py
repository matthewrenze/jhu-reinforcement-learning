import pytest
from unittest.mock import Mock, patch, mock_open
import numpy as np
from tiles.tile_factory import TileFactory
from tiles.test_tiles import TestTiles

@pytest.fixture
def setup():
    map = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    tiles = TestTiles.create([[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 3, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]])
    factory = TileFactory()
    return map, tiles, factory

def test_create(setup):
    map, tiles, factory = setup
    factory._load = Mock(return_value=map)
    factory._convert = Mock(return_value=tiles)
    tiles = factory.create(1, 0, False)
    actual_tiles = factory.create(1)
    factory._load.assert_called_with(1)
    factory._convert.assert_called_with(map)
    assert np.array_equal(tiles, actual_tiles)

@patch('builtins.open', new_callable=mock_open, read_data="test_map")
def test_load(open_mock, setup):
    map, _, factory = setup
    file = Mock()
    result = factory._load(1)
    open_mock.assert_called_with("../data/levels/level-1.txt", 'r')
    assert result == "test_map"

def test_convert(setup):
    map, tiles, factory = setup
    actual = factory._convert(map)
    assert np.array_equal(actual, tiles)

@pytest.mark.parametrize("rotation, expected", [
    (0, [[1, 2], [3, 4]]),
    (1, [[2, 4], [1, 3]]),
    (2, [[4, 3], [2, 1]]),
    (3, [[3, 1], [4, 2]])])
def test_rotation(setup, rotation, expected):
    _, _, factory = setup
    tiles = TestTiles.create([[1, 2], [3, 4]])
    factory._convert = Mock(return_value=tiles)
    expected = TestTiles.create(expected)
    actual = factory._rotate(tiles, rotation)
    assert np.array_equal(actual, expected)

def test_flip(setup):
    _, _, factory = setup
    tiles = TestTiles.create([[1, 2], [3, 4]])
    factory._convert = Mock(return_value=tiles)
    expected = TestTiles.create([[3, 4], [1, 2]])
    actual = factory._flip(tiles)
    assert np.array_equal(actual, expected)
