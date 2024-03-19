import pytest
from unittest.mock import Mock
from ghosts.ghost import Ghost
from ghosts.ghost import Mode
from tiles.tile import Tile
from actions.action import Action

@pytest.mark.parametrize("previous_mode, current_mode, expected", [
    (Mode.SCATTER, Mode.SCATTER, False),
    (Mode.SCATTER, Mode.CHASE, True),
    (Mode.CHASE, Mode.SCATTER, True),
    (Mode.CHASE, Mode.CHASE, False)])
def test_should_reverse(previous_mode, current_mode, expected):
    ghost = Ghost(Tile.BLINKY, (1, 1), (0, 0), Mock())
    assert ghost.should_reverse(previous_mode, current_mode) == expected

@pytest.mark.parametrize("orientation, expected", [
    (Action.UP, Action.DOWN),
    (Action.DOWN, Action.UP),
    (Action.LEFT, Action.RIGHT),
    (Action.RIGHT, Action.LEFT)])
def test_get_reverse(orientation, expected):
    ghost = Ghost(Tile.BLINKY, (1, 1), (0, 0), Mock())
    assert ghost.get_reverse(orientation) == expected

