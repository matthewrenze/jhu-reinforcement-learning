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
    state = [[0, 1, 2], [3, 4, 5], [0, 0, 0]]
    state = np.ndarray(shape=(3, 3), buffer=np.array(state), dtype=int)
    agent = TestAgent((0, 2))
    ghost = TestGhost((1, 2))
    environment = Environment(state, agent, [ghost])
    actual_state = environment.get_state()
    expected_state = np.array([[0, 1, 2], [3, 4, 5], [0, 0, 0]])
    assert np.array_equal(actual_state, expected_state)

# Create a parameterized test for the test_is_valid_move function
@pytest.mark.parametrize("new_location, expected", [
    ((0, 0), True),
    ((1, 2), False)])
def test_is_valid_move(new_location, expected):
    state = np.zeros((3, 3))
    state[1, 2] = Tile.WALL.id
    environment = Environment(state, TestAgent((1, 1)), [])
    actual = environment._is_valid_move(new_location)
    assert actual == expected

@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_game_over", [
    (Action.NONE.value, (1, 1, Tile.PACMAN.id), (1, 1, Tile.PACMAN.id), 0, False),
    (Action.UP.value, (1, 1, Tile.EMPTY.id), (0, 1, Tile.PACMAN.id), 0, False),
    (Action.DOWN.value, (1, 1, Tile.EMPTY.id), (2, 1, Tile.PACMAN.id), 0, True),
    (Action.LEFT.value, (1, 1, Tile.EMPTY.id), (1, 0, Tile.PACMAN.id), 10, True),
    (Action.RIGHT.value, (1, 1, Tile.PACMAN.id), (1, 2, Tile.WALL.id), 0, False)])
def test_execute_action(action, state_change_1, state_change_2, expected_reward, expected_game_over):
    state = [
        [Tile.EMPTY.id, Tile.EMPTY.id, Tile.EMPTY.id],
        [Tile.DOT.id, Tile.PACMAN.id, Tile.WALL.id],
        [Tile.EMPTY.id, Tile.STATIC.id, Tile.EMPTY.id]]
    state = np.ndarray(shape=(3, 3), buffer=np.array(state), dtype=int)
    expected_state = state.copy()
    expected_state[state_change_1[0]][state_change_1[1]] = state_change_1[2]
    expected_state[state_change_2[0]][state_change_2[1]] = state_change_2[2]
    environment = Environment(state, TestAgent((1, 1)), [TestGhost((2, 1))])
    environment._get_random_action = Mock(return_value=Action.NONE.value)
    actual_state, actual_reward, actual_game_over = environment.execute_action(action)
    assert np.array_equal(expected_state, actual_state)
    assert expected_reward == actual_reward
    assert expected_game_over == actual_game_over

def test_power_up_makes_pacman_invincible():
    state = [
        [Tile.PACMAN.id, Tile.POWER.id],
        [Tile.EMPTY.id, Tile.EMPTY.id]]
    state = np.ndarray(shape=(2, 2), buffer=np.array(state), dtype=int)
    environment = Environment(state, TestAgent(), [])
    assert environment.invincible_time == 0
    assert not environment._is_invincible()
    environment.execute_action(Action.RIGHT.value)
    assert environment.invincible_time == 25
    assert environment._is_invincible()
    environment.execute_action(Action.NONE.value)
    assert environment.invincible_time == 24
    assert environment._is_invincible()

def test_vulnerable_allows_ghosts_to_kill_pacman():
    state = [
        [Tile.PACMAN.id, Tile.STATIC.id],
        [Tile.DOT.id, Tile.EMPTY.id]]
    state = np.ndarray(shape=(2, 2), buffer=np.array(state), dtype=int)
    environment = Environment(state, TestAgent(), [TestGhost((1, 0))])
    environment._get_random_action = Mock(return_value=Action.NONE.value)
    environment.execute_action(Action.RIGHT.value)
    assert not environment._is_invincible()
    assert environment.is_game_over

def test_invincible_allows_pacman_to_eat_ghosts():
    state = [
        [Tile.PACMAN.id, Tile.STATIC.id],
        [Tile.DOT.id, Tile.EMPTY.id]]
    state = np.ndarray(shape=(2, 2), buffer=np.array(state), dtype=int)
    environment = Environment(state, TestAgent(), [TestGhost((0, 1))])
    environment._ghost_respawns = [TestGhost((1, 1))]
    environment._get_random_action = Mock(return_value=Action.NONE.value)
    environment.invincible_time = 25
    state, reward, is_game_over = environment.execute_action(Action.RIGHT.value)
    assert reward == 200
    assert not is_game_over
    assert state[0, 1] == Tile.PACMAN.id
    assert state[1, 1] == Tile.STATIC.id

@pytest.mark.parametrize("start_location, action, expected_location", [
    ((0, 1), Action.UP.value, (2, 1)),
    ((2, 1), Action.DOWN.value, (0, 1)),
    ((1, 0), Action.LEFT.value, (1, 2)),
    ((1, 2), Action.RIGHT.value, (1, 0))])
def test_teleport(start_location, action, expected_location):
    state = np.zeros((3, 3))
    environment = Environment(state, TestAgent(start_location), [])
    environment.execute_action(action)
    assert environment.agent.location == expected_location







