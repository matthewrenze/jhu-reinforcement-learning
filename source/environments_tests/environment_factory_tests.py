import numpy as np
import unittest
from unittest.mock import Mock
from environments.environment_factory import EnvironmentFactory

def test_create():
    file_data = "#  #  #  #  #\n#  .  m  .  #\n#  .  #  .  #\n#  .  c  .  #\n#  #  #  #  #"
    expected_state = [
        [1, 1, 1, 1, 1],
        [1, 4, 0, 4, 1],
        [1, 4, 1, 4, 1],
        [1, 4, 0, 4, 1],
        [1, 1, 1, 1, 1]]
    expected_state = np.array(expected_state)
    factory = EnvironmentFactory()
    factory._read_file = Mock()
    factory._read_file.return_value = file_data
    environment = factory.create(1)
    expected_agent_location = (3, 2)
    expected_state[expected_agent_location] = 0
    expected_ghost_locations = [(1, 2)]
    factory._read_file.assert_called_with("curriculum/level-1.txt")
    assert np.array_equal(expected_state, environment._state)
    assert expected_agent_location == environment._agent_location
    assert expected_ghost_locations == environment._ghost_locations


