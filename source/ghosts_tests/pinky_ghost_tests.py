import pytest
from ghosts.pinky_ghost import PinkyGhost
from tiles.tile import Tile
from actions.action import Action
from houses.house import House

@pytest.fixture
def setup():
    pinky = PinkyGhost((1, 1), House([(0, 0)], (1, 1)))
    return pinky

def test_init(setup):
    pinky = setup
    assert pinky.tile == Tile.PINKY
    assert pinky.scatter_target == (0, 4)
    assert pinky.wait_time == 4

@pytest.mark.parametrize("agent_orientation, expected", [
    (Action.UP.value, (-4, 0)),
    (Action.DOWN.value, (4, 0)),
    (Action.LEFT.value, (0, -4)),
    (Action.RIGHT.value, (0, 4))])
def test_get_chase_target(agent_orientation, expected, setup):
    pinky = setup
    agent_location = (0, 0)
    agent_orientation = Action.UP.value
    ghost_locations = [(Tile.PINKY.id, (1, 1))]
    actual = pinky.get_chase_target(agent_location, agent_orientation, ghost_locations)
    expected = (-4, 0)
    assert actual == expected