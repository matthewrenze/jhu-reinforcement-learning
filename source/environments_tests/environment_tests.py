import pytest
from unittest.mock import Mock
import numpy as np
from environments.environment import Environment
from tiles.tiles import Tiles
from tiles.test_tiles import TestTiles
from tiles.tile import Tile
from actions.action import Action
from agents.test_agent import TestAgent
from ghosts.test_ghost import TestGhost
from ghosts.ghost import Mode

def test_reset():
    tiles = TestTiles.create_zeros(3)
    agent = TestAgent((0, 0))
    environment = Environment(tiles, agent, [])
    with pytest.raises(NotImplementedError):
        environment.reset(1)

def test_get_state():
    tile_ints = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    expected_tiles = np.ndarray((3, 3), buffer=np.array(tile_ints), dtype=int)
    tiles = TestTiles.create(tile_ints)
    environment = Environment(tiles, TestAgent((0, 2)), [TestGhost((1, 2))])
    actual_state = environment.get_state()
    assert np.array_equal(actual_state.tiles, expected_tiles)
    assert actual_state.agent_location == (0, 2)
    assert actual_state.agent_orientation == Action.NONE.value
    assert actual_state.ghost_locations == [(Tile.STATIC.id, (1, 2))]
    assert not actual_state.is_invincible
    assert actual_state.ghost_mode == Mode.SCATTER.value


def test_is_invincible():
    tiles = TestTiles.create_zeros(3)
    environment = Environment(tiles, TestAgent(), [])
    assert not environment._is_invincible()
    environment._invincible_time = 1
    assert environment._is_invincible()

def test_decrement_invincible_time():
    tiles = TestTiles.create_zeros(3)
    environment = Environment(tiles, TestAgent(), [])
    environment._invincible_time = 1
    environment._decrement_invincible_time()
    assert environment._invincible_time == 0
    environment._decrement_invincible_time()
    assert environment._invincible_time == 0

@pytest.mark.parametrize("new_location, expected", [
    ((0, 0), True),
    ((1, 2), False)])
def test_is_valid_move(new_location, expected):
    tiles = TestTiles.create_zeros(3)
    environment = Environment(tiles, TestAgent((1, 1)), [])
    environment._tiles[1, 2] = Tile.WALL
    actual = environment._is_valid_move(new_location)
    assert actual == expected

def test_can_teleport():
    tiles = TestTiles.create_zeros(3)
    environment = Environment(tiles, TestAgent(), [])
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
    tiles = TestTiles.create_zeros(3)
    environment = Environment(tiles, TestAgent(), [])
    actual_location = environment._teleport(new_location)
    assert actual_location == expected_location

@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_invincible_time", [
    (Action.NONE, (1, 1, Tile.PACMAN), (1, 1, Tile.PACMAN), 0, 0),
    (Action.UP, (1, 1, Tile.EMPTY), (0, 1, Tile.PACMAN), 0, 0),
    (Action.DOWN, (1, 1, Tile.EMPTY), (2, 1, Tile.PACMAN), 50, 25),
    (Action.LEFT, (1, 1, Tile.EMPTY), (1, 0, Tile.PACMAN), 10, 0),
    (Action.RIGHT, (1, 1, Tile.PACMAN), (1, 2, Tile.WALL), 0, 0)])
def test_move_agent(action, state_change_1, state_change_2, expected_reward, expected_invincible_time):
    tiles = Tiles([
        [Tile.EMPTY, Tile.EMPTY, Tile.EMPTY],
        [Tile.DOT, Tile.EMPTY, Tile.WALL],
        [Tile.EMPTY, Tile.POWER, Tile.EMPTY]])
    expected_tiles = tiles.to_integer_array()
    expected_tiles[state_change_1[0]][state_change_1[1]] = state_change_1[2].id
    expected_tiles[state_change_2[0]][state_change_2[1]] = state_change_2[2].id
    environment = Environment(tiles, TestAgent((1, 1)), [])
    environment._move_agent(action)
    state = environment.get_state()
    assert np.array_equal(state.tiles, expected_tiles)
    assert environment.reward == expected_reward
    assert environment._invincible_time == expected_invincible_time

@pytest.mark.parametrize("tile, expected_is_game_over, expected_is_winner", [
    (Tile.DOT, False, False),
    (Tile.EMPTY, True, True)])
def test_check_if_level_complete(tile, expected_is_game_over, expected_is_winner):
    tiles = Tiles([
        [Tile.EMPTY, Tile.EMPTY],
        [Tile.EMPTY, tile]])
    environment = Environment(tiles, TestAgent(), [])
    environment._check_if_level_complete()
    assert environment.is_game_over == expected_is_game_over
    assert environment.is_winner == expected_is_winner

@pytest.mark.parametrize("ghost_location, is_invincible, expected_reward, expected_location, expected_is_game_over", [
    ((1, 1), False, 0, (1, 1), False),
    ((0, 0), False, 0, (0, 0), True),
    ((1, 1), True, 0, (1, 1), False),
    ((0, 0), True, 200, (1, 0), False)])
def test_check_if_ghosts_touching(ghost_location, is_invincible, expected_reward, expected_location, expected_is_game_over):
    tiles = TestTiles.create_zeros(2)
    environment = Environment(tiles, TestAgent(), [TestGhost(ghost_location)])
    environment._ghost_spawn_locations = [TestGhost((1, 0))]
    environment._is_invincible = Mock(return_value=is_invincible)
    environment._check_if_ghosts_touching()
    assert environment.reward == expected_reward
    assert environment.ghosts[0].location == expected_location
    assert environment.is_game_over == expected_is_game_over
    assert not environment.is_winner

@pytest.mark.parametrize("action, expected_location", [
    (Action.NONE, (1, 1)),
    (Action.UP, (0, 1)),
    (Action.DOWN, (2, 1)),
    (Action.LEFT, (1, 0)),
    (Action.RIGHT, (1, 2))])
def test_move_ghosts(action, expected_location):
    tiles = TestTiles.create_zeros(3)
    ghost = TestGhost((1, 1))
    environment = Environment(tiles, TestAgent(), [ghost])
    ghost.select_action = Mock(return_value=action)
    environment._move_ghosts()
    assert environment.ghosts[0].location == expected_location

# TODO: Should this be simplified since I'm testing each individual private method above?
@pytest.mark.parametrize("action, state_change_1, state_change_2, expected_reward, expected_game_over", [
    (Action.NONE, (1, 1, Tile.PACMAN), (1, 1, Tile.PACMAN), 0, False),
    (Action.UP, (1, 1, Tile.EMPTY), (0, 1, Tile.PACMAN), 0, False),
    (Action.DOWN, (1, 1, Tile.EMPTY), (2, 1, Tile.PACMAN), 0, True),
    (Action.LEFT, (1, 1, Tile.EMPTY), (1, 0, Tile.PACMAN), 10, True),
    (Action.RIGHT, (1, 1, Tile.PACMAN), (1, 2, Tile.WALL), 0, False)])
def test_execute_action(action, state_change_1, state_change_2, expected_reward, expected_game_over):
    tiles = Tiles([
        [Tile.EMPTY, Tile.EMPTY, Tile.EMPTY],
        [Tile.DOT, Tile.EMPTY, Tile.WALL],
        [Tile.EMPTY, Tile.STATIC, Tile.EMPTY]])
    expected_tiles = tiles.to_integer_array()
    expected_tiles[state_change_1[0]][state_change_1[1]] = state_change_1[2].id
    expected_tiles[state_change_2[0]][state_change_2[1]] = state_change_2[2].id
    environment = Environment(tiles, TestAgent((1, 1)), [TestGhost((2, 1))])
    actual_state, actual_reward, actual_game_over = environment.execute_action(action)
    assert np.array_equal(expected_tiles, actual_state.tiles)
    assert expected_reward == actual_reward
    assert expected_game_over == actual_game_over
