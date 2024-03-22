from states.state import State
from actions.action import Action

class Agent:

    # NOTE: Agent always starts with orientation to the right

    def __init__(self, location: tuple[int, int], hyperparameters: dict[str, float] = None):
        self.location = location
        self.orientation = Action.RIGHT
        self.hyperparameters = hyperparameters

    def select_action(self, state: State) -> Action:
        raise NotImplementedError("get_action method must be implemented in the subclass.")

    def update(self, state: State, action: Action, reward: int, next_state: State) -> None:
        pass

    def get_model(self) -> object:
        pass

    def set_model(self, model) -> None:
        pass
