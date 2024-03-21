import numpy as np
from unittest.mock import Mock, patch, mock_open

import pytest

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
    tiles = factory.create(1)
    factory._load.assert_called_with(1)
    factory._convert.assert_called_with(map)
    assert np.array_equal(tiles, tiles)


    # factory = TileFactory()
    # map = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    # expected_tiles = TestTiles.create([[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 3, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]])
    # tiles = factory.create(map)
    # assert np.array_equal(tiles, expected_tiles)

@patch('builtins.open', new_callable=mock_open, read_data="test_map")
def test_load(open_mock, setup):
    map, _, factory = setup
    file = Mock()
    result = factory._load(1)
    open_mock.assert_called_with("levels/level-1.txt", 'r')
    assert result == "test_map"

def test_convert(setup):
    map, tiles, factory = setup
    actual = factory._convert(map)
    assert np.array_equal(actual, tiles)
