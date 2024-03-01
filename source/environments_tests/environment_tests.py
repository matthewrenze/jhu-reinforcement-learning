import pytest
from unittest.mock import Mock
import numpy as np
from environments.environment import Environment
from tiles.tile import Tile
from agents.actions import Action
from agents.test_agent import TestAgent
from ghosts.test_ghost import TestGhost

def test_reset():
    state = np.zeros((10, 10))
    agent = TestAgent((0, 0))
    environment = Environment(state, agent, [])
    with pytest.raises(NotImplementedError):
        environment.reset(1)

def test_get_state():
    state = np.array([[0, 1, 2], [3, 4, 5], [0, 0, 0]])
    environment = Environment(state, TestAgent((0, 2)), [TestGhost((1, 2))])
    actual_state = environment.get_state()
    expected_state = np.array([[0, 1, 2], [3, 4, 5], [0, 0, 0]])
    assert np.array_equal(actual_state, expected_state)

def test_is_invincible():
    environment = Environment(np.zeros((3, 3)), TestAgent(), [])
    assert not environment._is_invincible()
    environment.invincible_time = 1
    assert environment._is_invincible()

def test_decrement_invincible_time():
    environment = Environment(np.zeros((3, 3)), TestAgent(), [])
    environment.invincible_time = 1
    environment._decrement_invincible_time()
    assert environment.invincible_time == 0
    environment._decrement_invincible_time()
    assert environment.invincible_time == 0

@pytest.mark.parametrize("new_location, expected", [
    ((0, 0), True),
    ((1, 2), False)])
def test_is_valid_move(new_location, expected):
    environment = Environment(np.zeros((3, 3)), TestAgent((1, 1)), [])
    environment._state[1, 2] = Tile.WALL.id
    actual = environment._is_valid_move(new_location)
    assert actual == expected

def test_can_teleport():
    environment = Environment(np.zeros((3, 3)), TestAgent(), [])
    assert environment._can_teleport((-1, 1))
    assert environment._can_teleport((3, 1))
    assert environment._can_teleport((1, -1))
    assert environment._can_teleport((1, 3))
    assert not environment._can_teleport((2, 2))

@pytest.mark.parametrize("new_location, action, expected_location", [
    ((-1, 1), Action.UP.value, (2, 1)),
    ((3, 1), Action.DOWN.value, (0, 1)),
    ((1, -1), Action.LEFT.value, (1, 2)),
    ((1, 3), Action.RIGHT.value, (1, 0))])
def test_teleport(new_location, action, expected_location):
    environment = Environment(np.zeros((3, 3)), TestAgent(), [])
    actual_location = environment._teleport(new_location)
    assert actual_location == expected_location

@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_invincible_time", [
    (Action.NONE.value, (1, 1, Tile.PACMAN.id), (1, 1, Tile.PACMAN.id), 0, 0),
    (Action.UP.value, (1, 1, Tile.EMPTY.id), (0, 1, Tile.PACMAN.id), 0, 0),
    (Action.DOWN.value, (1, 1, Tile.EMPTY.id), (2, 1, Tile.PACMAN.id), 50, 25),
    (Action.LEFT.value, (1, 1, Tile.EMPTY.id), (1, 0, Tile.PACMAN.id), 10, 0),
    (Action.RIGHT.value, (1, 1, Tile.PACMAN.id), (1, 2, Tile.WALL.id), 0, 0)])
def test_move_agent(action, state_change_1, state_change_2, expected_reward, expected_invincible_time):
    state = np.array([
        [Tile.EMPTY.id, Tile.EMPTY.id, Tile.EMPTY.id],
        [Tile.DOT.id, Tile.PACMAN.id, Tile.WALL.id],
        [Tile.EMPTY.id, Tile.POWER.id, Tile.EMPTY.id]])
    expected_state = state.copy()
    expected_state[state_change_1[0]][state_change_1[1]] = state_change_1[2]
    expected_state[state_change_2[0]][state_change_2[1]] = state_change_2[2]
    environment = Environment(state, TestAgent((1, 1)), [])
    environment._move_agent(action)
    assert np.array_equal(environment.get_state(), expected_state)
    assert environment.reward == expected_reward
    assert environment.invincible_time == expected_invincible_time

@pytest.mark.parametrize("tile, expected_is_game_over, expected_is_winner", [
    (Tile.DOT.id, False, False),
    (Tile.EMPTY.id, True, True)])
def test_check_if_level_complete(tile, expected_is_game_over, expected_is_winner):
    state = np.array([
        [Tile.PACMAN.id, Tile.EMPTY.id],
        [Tile.EMPTY.id, tile]])
    environment = Environment(state, TestAgent(), [])
    environment._check_if_level_complete()
    assert environment.is_game_over == expected_is_game_over
    assert environment.is_winner == expected_is_winner

@pytest.mark.parametrize("ghost_location, is_invincible, expected_reward, expected_location, expected_is_game_over", [
    ((1, 1), False, 0, (1, 1), False),
    ((0, 0), False, 0, (0, 0), True),
    ((1, 1), True, 0, (1, 1), False),
    ((0, 0), True, 200, (1, 0), False)])
def test_check_if_ghosts_touching(ghost_location, is_invincible, expected_reward, expected_location, expected_is_game_over):
    environment = Environment(np.zeros((2, 2)), TestAgent(), [TestGhost(ghost_location)])
    environment._ghost_respawns = [TestGhost((1, 0))]
    environment._is_invincible = Mock(return_value=is_invincible)
    environment._check_if_ghosts_touching()
    assert environment.reward == expected_reward
    assert environment.ghosts[0].location == expected_location
    assert environment.is_game_over == expected_is_game_over
    assert not environment.is_winner

@pytest.mark.parametrize("action, expected_location", [
    (Action.NONE.value, (1, 1)),
    (Action.UP.value, (0, 1)),
    (Action.DOWN.value, (2, 1)),
    (Action.LEFT.value, (1, 0)),
    (Action.RIGHT.value, (1, 2))])
def test_move_ghosts(action, expected_location):
    ghost = TestGhost((1, 1))
    environment = Environment(np.zeros((3, 3)), TestAgent(), [ghost])
    ghost.select_action = Mock(return_value=action)
    environment._move_ghosts()
    assert environment.ghosts[0].location == expected_location

# TODO: Should this be simplified since I'm testing each individual private method above?
@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_game_over", [
    (Action.NONE.value, (1, 1, Tile.PACMAN.id), (1, 1, Tile.PACMAN.id), 0, False),
    (Action.UP.value, (1, 1, Tile.EMPTY.id), (0, 1, Tile.PACMAN.id), 0, False),
    (Action.DOWN.value, (1, 1, Tile.EMPTY.id), (2, 1, Tile.PACMAN.id), 0, True),
    (Action.LEFT.value, (1, 1, Tile.EMPTY.id), (1, 0, Tile.PACMAN.id), 10, True),
    (Action.RIGHT.value, (1, 1, Tile.PACMAN.id), (1, 2, Tile.WALL.id), 0, False)])
def test_execute_action(action, state_change_1, state_change_2, expected_reward, expected_game_over):
    state = np.array([
        [Tile.EMPTY.id, Tile.EMPTY.id, Tile.EMPTY.id],
        [Tile.DOT.id, Tile.PACMAN.id, Tile.WALL.id],
        [Tile.EMPTY.id, Tile.STATIC.id, Tile.EMPTY.id]])
    expected_state = state.copy()
    expected_state[state_change_1[0]][state_change_1[1]] = state_change_1[2]
    expected_state[state_change_2[0]][state_change_2[1]] = state_change_2[2]
    environment = Environment(state, TestAgent((1, 1)), [TestGhost((2, 1))])
    actual_state, actual_reward, actual_game_over = environment.execute_action(action)
    assert np.array_equal(expected_state, actual_state)
    assert expected_reward == actual_reward
    assert expected_game_over == actual_game_over







