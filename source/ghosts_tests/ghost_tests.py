import pytest
import unittest
import numpy as np
from unittest.mock import Mock, patch
from ghosts.ghost import Ghost
from houses.house import House
from ghosts.ghost import Mode
from tiles.tile import Tile
from states.state import State
from actions.action import Action

@pytest.fixture
def setup():
    tiles = np.zeros((3, 3))
    state = State(tiles, (0, 0), Action.NONE.value, [], False, Mode.SCATTER.value)
    house = House([(0, 0)], (3, 3))
    ghost = Ghost(Tile.STATIC, (1, 1), (2, 2), house)
    return tiles, state, house, ghost

def test_ghost_init(setup):
    _, _, _, ghost = setup
    assert ghost.tile == Tile.STATIC
    assert ghost.location == (1, 1)
    assert ghost.orientation == Action.NONE
    assert ghost.spawn_location == (1, 1)
    assert ghost.house_locations == [(0, 0)]
    assert ghost.house_exit_target == (3, 3)
    assert ghost.scatter_target == (2, 2)
    assert ghost.mode == Mode.SCATTER

@pytest.mark.parametrize("should_reverse, is_in_house, mode, expected_target", [
    (True, False, Mode.SCATTER, (None, None)),
    (False, True, Mode.SCATTER, (3, 3)),
    (False, False, Mode.CHASE, (0, 0)),
    (False, False, Mode.FRIGHTENED, (9, 9)),
    (False, False, Mode.SCATTER, (2, 2))])
def test_select_action(setup, should_reverse, is_in_house, mode, expected_target):
    tiles, state, _, ghost = setup
    state.ghost_mode = mode.value
    ghost.should_reverse = Mock(return_value=should_reverse)
    ghost.is_in_house = Mock(return_value=is_in_house)
    ghost.get_reverse = Mock(return_value=Action.NONE)
    ghost.find_best_move = Mock(return_value=(0, 0))
    ghost.get_chase_target = Mock(return_value=(0, 0))
    ghost.get_random_target = Mock(return_value=(9, 9))
    ghost.select_action(state)
    if (should_reverse):
        ghost.get_reverse.assert_called_with(Action.NONE)
    else:
        ghost.find_best_move.assert_called_with(tiles, expected_target)

@pytest.mark.parametrize("previous_mode, current_mode, expected", [
    (Mode.SCATTER, Mode.SCATTER, False),
    (Mode.SCATTER, Mode.CHASE, True),
    (Mode.CHASE, Mode.SCATTER, True),
    (Mode.CHASE, Mode.CHASE, False)])
def test_should_reverse(setup, previous_mode, current_mode, expected):
    _, _, _, ghost = setup
    assert ghost.should_reverse(previous_mode, current_mode) == expected

@pytest.mark.parametrize("orientation, expected", [
    (Action.UP, Action.DOWN),
    (Action.DOWN, Action.UP),
    (Action.LEFT, Action.RIGHT),
    (Action.RIGHT, Action.LEFT)])
def test_get_reverse(setup, orientation, expected):
    _, _, _, ghost = setup
    assert ghost.get_reverse(orientation) == expected

def test_is_in_house(setup):
    _, _, _, ghost = setup
    assert ghost.is_in_house((0, 0))
    assert not ghost.is_in_house((1, 1))

def test_get_chase_target(setup):
    _, _, _, ghost = setup
    with pytest.raises(NotImplementedError):
        ghost.get_chase_target((0, 0), Action.UP.value, [(Tile.STATIC.id, (1, 1))])

def test_get_random_target(monkeypatch, setup):
    _, _, _, ghost = setup
    monkeypatch.setattr(np.random, "randint", lambda x, y: 9)
    target = ghost.get_random_target(np.zeros((3, 3)))
    assert target == (9, 9)

def find_best_move(setup):
    tiles, _, _, ghost = setup
    ghost.tiles[0, 1] = Tile.WALL
    action = ghost.find_best_move(tiles, (1, 1))
    assert action == Action.LEFT

@pytest.mark.parametrize("action, expected_location", [
    (Action.UP, (0, 1)),
    (Action.DOWN, (2, 1)),
    (Action.LEFT, (1, 0)),
    (Action.RIGHT, (1, 2))])
def test_get_new_location(setup, action, expected_location):
    _, _, _, ghost = setup
    actual_location = ghost.get_new_location((1, 1), action)
    assert actual_location == expected_location

@pytest.mark.parametrize("is_reverse, is_wall, expected_is_valid", [
    (True, False, False),
    (False, True, False),
    (False, False, True)])
def test_is_valid_move(setup, is_reverse, is_wall, expected_is_valid):
    tiles, _, _, ghost = setup
    ghost.is_reverse = Mock(return_value=is_reverse)
    ghost.is_wall = Mock(return_value=is_wall)
    actual_is_valid = ghost.is_valid_move(tiles, (1, 1), Action.NONE)
    assert actual_is_valid == expected_is_valid

def test_is_reverse(setup):
    _, _, _, ghost = setup
    ghost.orientation = Action.UP
    assert not ghost.is_reverse(Action.NONE)
    assert not ghost.is_reverse(Action.UP)
    assert ghost.is_reverse(Action.DOWN)

def test_is_wall(setup):
    tiles, _, _, ghost = setup
    assert not ghost.is_wall(tiles, (1, 1))
    tiles[1, 1] = Tile.WALL.id
    assert ghost.is_wall(tiles, (1, 1))

def test_calculate_distance(setup):
    _, _, _, ghost = setup
    distance = ghost.calculate_distance((0, 0), (3, 4))
    assert distance == 5.0


