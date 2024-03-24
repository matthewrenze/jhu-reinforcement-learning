import numpy as np
import pytest
from environments.environment_factory import EnvironmentFactory
from agents.test_agent import TestAgent
from ghosts.test_ghost import TestGhost
from tiles.test_tiles import TestTiles
from tiles.tile import Tile

@pytest.fixture()
def setup():
    tiles = TestTiles.create_zeros(3)
    agent = TestAgent()
    ghosts = [TestGhost((1, 1))]
    factory = EnvironmentFactory()
    return tiles, agent, ghosts, factory

def test_create(setup):
    tiles, agent, ghosts, factory = setup
    environment = factory.create(tiles, agent, ghosts)
    assert np.array_equal(environment._tiles, tiles)
    assert environment.agent == agent
    assert environment.ghosts == ghosts

def test_clear_agents(setup):
    tiles, agent, ghosts, factory = setup
    tiles[0, 0] = Tile.PACMAN
    tiles[1, 1] = Tile.STATIC
    factory._clear_agents(tiles, agent, ghosts)
    assert (tiles == Tile.EMPTY).all()








