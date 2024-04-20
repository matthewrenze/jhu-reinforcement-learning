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
    ghost = Ghost((1, 1), house)
    ghost.tile = Tile.STATIC
    ghost.scatter_target = (2, 2)
    ghost.spawn_location = (4, 4)
    ghost.wait_time = 0
    return tiles, state, house, ghost

def test_ghost_init(setup):
    _, _, _, ghost = setup
    assert ghost.tile == Tile.STATIC
    assert ghost.location == (1, 1)
    assert ghost.orientation == Action.NONE
    assert ghost.spawn_location == (4, 4)
    assert ghost.house_locations == [(0, 0)]
    assert ghost.house_exit_target == (3, 3)
    assert ghost.scatter_target == (2, 2)
    assert ghost.mode == Mode.SCATTER

@pytest.mark.parametrize("wait_time, should_reverse, is_in_house, mode, expected_target", [
    (1, True, False, Mode.SCATTER, (None, None)),
    (0, True, False, Mode.SCATTER, (None, None)),
    (0, False, True, Mode.SCATTER, (3, 3)),
    (0, False, False, Mode.CHASE, (0, 0)),
    (0, False, False, Mode.FRIGHTENED, (9, 9)),
    (0, False, False, Mode.SCATTER, (2, 2))])
def test_select_action(setup, wait_time, should_reverse, is_in_house, mode, expected_target):
    tiles, state, _, ghost = setup
    state.ghost_mode = mode.value
    ghost._should_reverse = Mock(return_value=should_reverse)
    ghost._is_in_house = Mock(return_value=is_in_house)
    ghost._get_reverse = Mock(return_value=Action.NONE)
    ghost._find_best_move = Mock(return_value=(0, 0))
    ghost._get_chase_target = Mock(return_value=(0, 0))
    ghost._get_random_action = Mock(return_value=Action.NONE)
    action = ghost.select_action(state)
    if wait_time > 0:
        assert action == Action.NONE
    elif should_reverse:
        ghost._get_reverse.assert_called_with(Action.NONE)
    elif mode == Mode.FRIGHTENED:
        ghost._get_random_action.assert_called_with(tiles)
    else:
        ghost._find_best_move.assert_called_with(tiles, expected_target)

def test_on_eaten(setup):
    _, _, _, ghost = setup
    ghost.on_eaten()
    assert ghost.location == (4, 4)
    assert ghost.orientation == Action.NONE
    assert ghost.wait_time == 5

@pytest.mark.parametrize("previous_mode, current_mode, expected", [
    (Mode.SCATTER, Mode.SCATTER, False),
    (Mode.SCATTER, Mode.CHASE, True),
    (Mode.CHASE, Mode.SCATTER, True),
    (Mode.CHASE, Mode.CHASE, False)])
def test_should_reverse(setup, previous_mode, current_mode, expected):
    _, _, _, ghost = setup
    assert ghost._should_reverse(previous_mode, current_mode) == expected

@pytest.mark.parametrize("orientation, expected", [
    (Action.UP, Action.DOWN),
    (Action.DOWN, Action.UP),
    (Action.LEFT, Action.RIGHT),
    (Action.RIGHT, Action.LEFT)])
def test_get_reverse(setup, orientation, expected):
    _, _, _, ghost = setup
    assert ghost._get_reverse(orientation) == expected

def test_is_in_house(setup):
    _, _, _, ghost = setup
    assert ghost._is_in_house((0, 0))
    assert not ghost._is_in_house((1, 1))

def test_get_chase_target(setup):
    _, _, _, ghost = setup
    with pytest.raises(NotImplementedError):
        ghost._get_chase_target((0, 0), Action.UP.value, [(Tile.STATIC.id, (1, 1))])

@pytest.mark.parametrize("random_action_id, expected_action", [
    (Action.UP.value, Action.DOWN),
    (Action.DOWN.value, Action.DOWN),])
def test_get_random_action(setup, random_action_id, expected_action, monkeypatch):
    tiles, _, _, ghost = setup
    monkeypatch.setattr(np.random, "randint", Mock(side_effect=[random_action_id, 2]))
    tiles[0, 1] = Tile.WALL.id
    action = ghost._get_random_action(tiles)
    assert action == expected_action

@pytest.mark.parametrize("final_tile, expected_is_dead_end", [
    (Tile.WALL.id, True),
    (Tile.EMPTY.id, False)])
def test_is_dead_end(setup, final_tile, expected_is_dead_end):
    tiles = np.array([[1, 0, 1], [1, 9, 1], [1, 1, 1]])
    tiles[2, 1] = final_tile
    ghost = Ghost((1, 1), House([], (0, 0)))
    assert ghost._is_dead_end(tiles) == expected_is_dead_end

@pytest.mark.parametrize("exit_location, expected_action", [
    ((0, 1), Action.UP),
    ((2, 1), Action.DOWN),
    ((1, 0), Action.LEFT),
    ((1, 2), Action.RIGHT)])
def test_exit_dead_end(setup, exit_location, expected_action):
    tiles = np.array([[1, 1, 1], [1, 9, 1], [1, 1, 1]])
    tiles[exit_location] = Tile.EMPTY.id
    ghost = Ghost((1, 1), House([], (0, 0)))
    action = ghost._exit_dead_end(tiles)
    assert action == expected_action

def find_best_move(setup):
    tiles, _, _, ghost = setup
    tiles[0, 1] = Tile.WALL.id
    action = ghost._find_best_move(tiles, (1, 1))
    assert action == Action.LEFT

@pytest.mark.parametrize("action, expected_location", [
    (Action.UP, (0, 1)),
    (Action.DOWN, (2, 1)),
    (Action.LEFT, (1, 0)),
    (Action.RIGHT, (1, 2))])
def test_get_new_location(setup, action, expected_location):
    tiles, _, _, ghost = setup
    actual_location = ghost._get_new_location((1, 1), action, tiles)
    assert actual_location == expected_location

@pytest.mark.parametrize("is_reverse, is_wall, expected_is_valid", [
    (True, False, False),
    (False, True, False),
    (False, False, True)])
def test_is_valid_move(setup, is_reverse, is_wall, expected_is_valid):
    tiles, _, _, ghost = setup
    ghost._is_reverse = Mock(return_value=is_reverse)
    ghost._is_wall = Mock(return_value=is_wall)
    actual_is_valid = ghost._is_valid_move(tiles, (1, 1), Action.NONE)
    assert actual_is_valid == expected_is_valid

def test_is_reverse(setup):
    _, _, _, ghost = setup
    ghost.orientation = Action.UP
    assert not ghost._is_reverse(Action.NONE)
    assert not ghost._is_reverse(Action.UP)
    assert ghost._is_reverse(Action.DOWN)

def test_is_wall(setup):
    tiles, _, _, ghost = setup
    assert not ghost._is_wall(tiles, (1, 1))
    tiles[1, 1] = Tile.WALL.id
    assert ghost._is_wall(tiles, (1, 1))

def test_calculate_distance(setup):
    _, _, _, ghost = setup
    distance = ghost._calculate_distance((0, 0), (3, 4))
    assert distance == 5.0


