import numpy as np
from environments.environment_factory import EnvironmentFactory


# TODO: Create a test environment specifically to test all the tiles
def test_create():
    actual_state = [
        [1, 1, 1, 1, 1],
        [1, 4, 0, 4, 1],
        [1, 4, 1, 4, 1],
        [1, 4, 0, 4, 1],
        [1, 1, 1, 1, 1]]
    actual_state = np.array(actual_state)
    factory = EnvironmentFactory()
    environment = factory.create(2)
    expected_state = np.array(actual_state)
    expected_agent_location = (3, 2)
    expected_state[expected_agent_location] = 0
    expected_ghost_locations = [(1, 2)]
    assert np.array_equal(expected_state, environment._state)
    assert expected_agent_location == environment._agent_location
    assert expected_ghost_locations == environment._ghost_locations


