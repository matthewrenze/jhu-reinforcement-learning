from states.state import State
from actions.action import Action

class Agent:

    # NOTE: Agent always starts with orientation to the right

    def __init__(self, location: tuple[int, int]):
        self.location = location
        self.orientation = Action.RIGHT

    def select_action(self, state: State) -> Action:
        raise NotImplementedError("get_action method must be implemented in the subclass.")