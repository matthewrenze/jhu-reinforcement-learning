import numpy as np
from tiles.tile import Tile
from agents.actions import Action

class Ghost:

    def __init__(self, tile: Tile, location: tuple[int, int]):
        self.tile = tile
        self.location = location

    def select_action(self, state: np.ndarray, agent_location: tuple[int, int], is_scattering: bool) -> int:
        # TODO: Remove this default action once we implement the select_action method for each ghost
        return Action.NONE.value
        # raise NotImplementedError("You must implement the select_action method in your subclass")