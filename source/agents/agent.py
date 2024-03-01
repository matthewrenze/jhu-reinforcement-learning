import numpy as np

class Agent:

    def __init__(self, location: tuple[int, int]):
        self.location = location

    def select_action(self, state: np.ndarray) -> int:
        raise NotImplementedError("get_action method must be implemented in the subclass.")