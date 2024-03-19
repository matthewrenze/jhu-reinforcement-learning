import pytest
from ghosts.clyde_ghost import ClydeGhost
from houses.house import House
from tiles.tile import Tile

@pytest.fixture
def setup():
    clyde = ClydeGhost((0, 0), House([(0, 0)], (1, 1)))
    return clyde

def test_init(setup):
    clyde = setup
    assert clyde.tile == Tile.CLYDE
    assert clyde.scatter_target == (20, 0)
    assert clyde.wait_time == 40

@pytest.mark.parametrize("agent_location, expected_target", [
    ((0, 8), (20, 0)),
    ((0, 9), (0, 0)),
    ((8, 0), (20, 0)),
    ((9, 0), (0, 0)),
    ((7, 7), (20, 0)),
    ((9, 9), (0, 0))])
def test_get_chase_target(setup, agent_location, expected_target):
    clyde = setup
    agent_location = (0, 8)
    agent_orientation = 1
    ghost_locations = [(Tile.CLYDE.id, (4, 0))]
    expected_target = (20, 0)
    actual_target = clyde._get_chase_target(agent_location, agent_orientation, ghost_locations)
    assert actual_target == expected_target
