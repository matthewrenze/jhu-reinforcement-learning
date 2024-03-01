import numpy as np
from unittest.mock import Mock
from environments.environment_factory import EnvironmentFactory
from agents.test_agent import TestAgent
from ghosts.test_ghost import TestGhost

def test_load():
    factory = EnvironmentFactory(Mock(), Mock())
    factory._read_file = Mock()
    factory._load(1)
    factory._read_file.assert_called_with("levels/level-1.txt")

def test_convert():
    factory = EnvironmentFactory(Mock(), Mock())
    environment = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    expected_state = [
        [1, 1, 1, 1, 1],
        [1, 3, 5, 3, 1],
        [1, 3, 1, 3, 1],
        [1, 3, 2, 3, 1],
        [1, 1, 1, 1, 1]]
    expected_state = np.array(expected_state)
    actual_state = factory._convert(environment)
    assert np.array_equal(actual_state, expected_state)

def test_create():
    # TODO: Simplify this test
    file_data = "#  #  #  #  #\n#  .  s  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    expected_state_1 = np.array([
        [1, 1, 1, 1, 1],
        [1, 3, 5, 3, 1],
        [1, 3, 1, 3, 1],
        [1, 3, 2, 3, 1],
        [1, 1, 1, 1, 1]])
    expected_state_2 = np.array([
        [1, 1, 1, 1, 1],
        [1, 3, 5, 3, 1],
        [1, 3, 1, 3, 1],
        [1, 3, 2, 3, 1],
        [1, 1, 1, 1, 1]])
    expected_agent = TestAgent((3, 2))
    expected_ghosts = [TestGhost((1, 2))]
    agent_factory = Mock()
    ghost_factory = Mock()
    factory = EnvironmentFactory(agent_factory, ghost_factory)
    factory._load = Mock(return_value=file_data)
    factory._convert = Mock(return_value=expected_state_1)
    agent_factory.create = Mock(return_value=expected_agent)
    ghost_factory.create = Mock(return_value=expected_ghosts)
    environment = factory.create(1, "pacman")
    factory._load.assert_called_with(1)
    factory._convert.assert_called_with(file_data)
    agent_factory.create.assert_called_with("pacman", expected_state_1)
    ghost_factory.create.assert_called_with(expected_state_1)
    assert np.array_equal(environment.get_state(), expected_state_2)
    assert environment.agent == expected_agent
    assert environment.ghosts == expected_ghosts




