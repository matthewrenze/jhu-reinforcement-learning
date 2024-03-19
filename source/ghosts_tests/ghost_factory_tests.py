import numpy as np
from ghosts.ghost_factory import GhostFactory
from houses.house import House
from tiles.tile import Tile
import pytest

def test_get_ghost_locations():
    state = np.array([
        [Tile.EMPTY.id, Tile.PACMAN.id],
        [Tile.STATIC.id, Tile.BLINKY.id]])
    factory = GhostFactory()
    actual = factory._get_ghosts(state)
    expected = [(Tile.STATIC, 1, 0), (Tile.BLINKY, 1, 1)]
    assert actual == expected

@pytest.mark.parametrize("tile, location, expected_type", [
    (Tile.STATIC, (2, 3), "StaticGhost"),
    (Tile.BLINKY, (2, 3), "BlinkyGhost"),
    (Tile.PINKY, (2, 3), "PinkyGhost"),
    (Tile.INKY, (2, 3), "InkyGhost"),
    (Tile.CLYDE, (2, 3), "ClydeGhost")])
def test_create_static_ghost(tile, location, expected_type):
    factory = GhostFactory()
    ghost = factory._create_ghost(Tile.STATIC, (2, 3), House([(0, 0)], (1, 1)))
    assert type(ghost).__name__ == "StaticGhost"
    assert ghost.location == (2, 3)
    assert ghost.spawn_location == (2, 3)

def test_create():
    factory = GhostFactory()
    tiles = [Tile.BLINKY.id, Tile.PINKY.id,
             Tile.INKY.id, Tile.CLYDE.id]
    tiles = np.ndarray(shape=(2, 2), buffer=np.array(tiles), dtype=int)
    house = House([(0, 0)], (1, 1))
    ghosts = factory.create(tiles, house)
    assert len(ghosts) == 4
    assert type(ghosts[0]).__name__ == "BlinkyGhost"
    assert type(ghosts[1]).__name__ == "PinkyGhost"
    assert type(ghosts[2]).__name__ == "InkyGhost"
    assert type(ghosts[3]).__name__ == "ClydeGhost"
    assert ghosts[0].location == (0, 0)
    assert ghosts[1].location == (0, 1)
    assert ghosts[2].location == (1, 0)
    assert ghosts[3].location == (1, 1)
    assert ghosts[0].house_locations == [(0, 0)]
    assert ghosts[0].house_exit_target == (1, 1)