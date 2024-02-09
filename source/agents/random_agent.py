import numpy as np
from agents.agent import Agent

# Note: This method is necessary to mocking the random action selection
def get_random_action():
    return np.random.choice(5)

class RandomAgent(Agent):

    def select_action(self, state: np.ndarray) -> int:
        return get_random_action()