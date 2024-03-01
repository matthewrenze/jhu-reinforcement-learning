import numpy as np
from agents.agent import Agent
from agents.actions import Action

class TestAgent(Agent):

    def __init__(self, location: tuple[int, int] = (0, 0)):
        super().__init__(location)

    def select_action(self, state: np.ndarray):
        return Action.NONE.value
