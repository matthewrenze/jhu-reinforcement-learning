import pytest
import numpy as np
from unittest.mock import Mock
from environments.environment_factory import EnvironmentFactory
from agents.test_agent import TestAgent
from ghosts.test_ghost import TestGhost
from houses.house import House
from tiles.tiles import Tiles

@pytest.fixture()
def data():
    map = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    expected_tiles = Tiles([[1, 1, 1, 1, 1], [1, 3, 5, 3, 1], [1, 3, 1, 3, 1], [1, 3, 2, 3, 1], [1, 1, 1, 1, 1]])
    expected_agent = TestAgent((3, 2))
    expected_ghosts = [TestGhost((1, 2))]
    expected_house = House([(1, 2)], (3, 4))
    hyperparameters = {"alpha": 0.1, "gamma": 0.2, "epsilon": 0.3}
    return map, expected_tiles, expected_agent, expected_ghosts, expected_house, hyperparameters

@pytest.fixture()
def setup(data):
    map, expected_tiles, expected_agent, expected_ghosts, expected_house, hyperparameters = data
    tile_factory = Mock()
    agent_factory = Mock()
    house_factory = Mock()
    ghost_factory = Mock()
    factory = EnvironmentFactory(tile_factory, agent_factory, house_factory, ghost_factory)
    factory._load = Mock(return_value=map)
    tile_factory.create = Mock(return_value=expected_tiles)
    agent_factory.create = Mock(return_value=expected_agent)
    ghost_factory.create = Mock(return_value=expected_ghosts)
    house_factory.create = Mock(return_value=expected_house)
    environment = factory.create(1, "pacman", hyperparameters)
    return tile_factory, agent_factory, house_factory, ghost_factory, factory, environment

def test_create(setup, data):
    tile_factory, agent_factory, house_factory, ghost_factory, factory, environment = setup
    map, expected_tiles, expected_agent, expected_ghosts, expected_house, hyperparameters = data
    factory._load.assert_called_with(1)
    assert tile_factory.create.called_with(map)
    assert agent_factory.create.called_with("pacman", expected_tiles)
    assert np.array_equal(environment._tiles, expected_tiles)
    assert environment.agent == expected_agent
    assert environment.ghosts == expected_ghosts

def test_load():
    factory = EnvironmentFactory(Mock(), Mock(), Mock(), Mock())
    factory._read_file = Mock()
    factory._load(1)
    factory._read_file.assert_called_with("levels/level-1.txt")




