import pytest
import numpy as np
from environments.states import Tile
from environments.states import get_agent_location
from environments.states import get_ghost_locations

@pytest.mark.parametrize("id, expected_tile", [
    (0, Tile.EMPTY),
    (1, Tile.WALL),
    (2, Tile.PACMAN),
    (3, Tile.DOT),
    (4, Tile.POWER),
    (5, Tile.BONUS),
    (6, Tile.GHOST)])
def test_get_enum_from_id(id, expected_tile):
    tile = Tile.get_enum_from_id(id)
    assert tile == expected_tile

@pytest.mark.parametrize("symbol, expected_tile", [
    (' ', Tile.EMPTY),
    ('#', Tile.WALL),
    ('c', Tile.PACMAN),
    ('m', Tile.GHOST),
    ('.', Tile.DOT),
    ('o', Tile.POWER),
    ('$', Tile.BONUS)])
def test_get_enum_from_symbol(symbol, expected_tile):
    tile = Tile.get_enum_from_symbol(symbol)
    assert tile == expected_tile

@pytest.mark.parametrize("id, expected_symbol", [
    (0, ' '),
    (1, '#'),
    (2, 'c'),
    (3, '.'),
    (4, 'o'),
    (5, '$'),
    (6, 'm')])
def test_get_symbol_from_id(id, expected_symbol):
    symbol = Tile.get_symbol_from_id(id)
    assert symbol == expected_symbol

@pytest.mark.parametrize("symbol, expected_id", [
    (' ', 0),
    ('#', 1),
    ('c', 2),
    ('.', 3),
    ('o', 4),
    ('$', 5),
    ('m', 6)])
def test_get_id_from_symbol(symbol, expected_id):
    id = Tile.get_id_from_symbol(symbol)
    assert id == expected_id

def test_get_agent_location():
    state = np.array([[0, 1], [2, 3]])
    agent_location = get_agent_location(state)
    expected_location = (1, 0)
    assert agent_location == expected_location

def test_get_ghost_locations():
    state = np.array([[0, 2], [6, 6]])
    ghost_locations = get_ghost_locations(state)
    expected_locations = [(1, 0), (1, 1)]
    assert ghost_locations == expected_locations