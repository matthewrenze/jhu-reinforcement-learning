import numpy as np
from agents.deep_q_learning_agent import DeepQLearningAgent
from states.state import State

# def test_convert_state():
#     tiles = np.array([[0, 1], [2, 3]])
#     agent_location = (1, 1)
#     ghost_locations = [(6, (0, 0)), (7, (0, 1)), (8, (0, 2))]
#     state = State(tiles, agent_location, 0, ghost_locations, True, 0)
#     agent = DeepQLearningAgent((1, 1), {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1})
#     flat_state = agent._convert_state(state)
#     assert len(flat_state) == 251
#     assert flat_state[0] == 1
#     assert flat_state[1] == 0
#     assert flat_state[26] == 1
#     assert flat_state[55] == 1
#     assert flat_state[81] == 1
#     assert flat_state[250] == 1


# def test_convert_state_manual():
#     tiles = np.array([[1, 1, 1, 1, 1], [1, 0, 3, 0, 1], [1, 3, 2, 3, 1], [1, 0, 3, 0, 1], [1, 1, 1, 1, 1]])
#     agent_location = (2, 3)
#     ghost_locations = [] #[(6, (0, 0)), (7, (0, 1)), (8, (0, 2))]
#     state = State(tiles, agent_location, 0, ghost_locations, True, 0)
#     agent = DeepQLearningAgent((2, 2), {"alpha": 0.1, "gamma": 0.9, "epsilon": 0.1})
#     flat_state = agent._convert_state(state)
#     # Print the flat state as 10  5 x 5 grids
#     for i in range(10):
#         for j in range(5):
#             print(flat_state[i * 25 + j * 5: i * 25 + j * 5 + 5])
#         print()

