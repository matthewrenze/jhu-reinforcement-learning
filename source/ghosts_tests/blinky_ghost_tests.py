import pytest
from ghosts.blinky_ghost import BlinkyGhost
from houses.house import House
from tiles.tile import Tile

@pytest.fixture
def setup():
    blinky = BlinkyGhost((0, 0), House([(0, 0)], (1, 1)))
    return blinky

def test_init(setup):
    blinky = setup
    assert blinky.tile == Tile.BLINKY
    assert blinky.scatter_target == (0, 16)
    assert blinky.wait_time == 0
