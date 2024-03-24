import pytest
from ghosts.inky_ghost import InkyGhost
from houses.house import House
from tiles.tile import Tile
from actions.action import Action

@pytest.fixture
def setup():
    inky = InkyGhost((0, 0), House([(0, 0)], (1, 1)))
    return inky

def test_init(setup):
    inky = setup
    assert inky.tile == Tile.INKY
    assert inky.scatter_target == (20, 20)
    assert inky.wait_time == 20

def test_get_chase_target(setup):
    inky = setup
    agent_location = (2, 0)
    agent_orientation = Action.RIGHT.value
    ghost_locations = [(Tile.BLINKY.id, (4, 0))]
    expected_target = (0, 4)
    actual_target = inky._get_chase_target(agent_location, agent_orientation, ghost_locations)
    assert actual_target == expected_target