import pytest
from tiles.tile import Tile

@pytest.mark.parametrize("id, expected_tile", [
    (0, Tile.EMPTY),
    (1, Tile.WALL),
    (2, Tile.PACMAN),
    (3, Tile.DOT),
    (4, Tile.POWER),
    (5, Tile.STATIC),
    (6, Tile.BLINKY),
    (7, Tile.PINKY),
    (8, Tile.INKY),
    (9, Tile.CLYDE)])
def test_get_enum_from_id(id, expected_tile):
    tile = Tile.get_enum_from_id(id)
    assert tile == expected_tile

@pytest.mark.parametrize("symbol, expected_tile", [
    (' ', Tile.EMPTY),
    ('#', Tile.WALL),
    ('c', Tile.PACMAN),
    ('.', Tile.DOT),
    ('o', Tile.POWER),
    ('s', Tile.STATIC),
    ('b', Tile.BLINKY),
    ('p', Tile.PINKY),
    ('i', Tile.INKY),
    ('y', Tile.CLYDE)
])
def test_get_enum_from_symbol(symbol, expected_tile):
    tile = Tile.get_enum_from_symbol(symbol)
    assert tile == expected_tile

@pytest.mark.parametrize("id, expected_symbol", [
    (0, ' '),
    (1, '#'),
    (2, 'c'),
    (3, '.'),
    (4, 'o'),
    (5, 's'),
    (6, 'b'),
    (7, 'p'),
    (8, 'i'),
    (9, 'y')])
def test_get_symbol_from_id(id, expected_symbol):
    symbol = Tile.get_symbol_from_id(id)
    assert symbol == expected_symbol

@pytest.mark.parametrize("symbol, expected_id", [
    (' ', 0),
    ('#', 1),
    ('c', 2),
    ('.', 3),
    ('o', 4),
    ('s', 5),
    ('b', 6),
    ('p', 7),
    ('i', 8),
    ('y', 9)])
def test_get_id_from_symbol(symbol, expected_id):
    id = Tile.get_id_from_symbol(symbol)
    assert id == expected_id
