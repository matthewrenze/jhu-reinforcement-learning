import numpy as np
from ghosts import Ghost
from agents.actions import Action
from tiles.tile import Tile

class StaticGhost(Ghost):

    def __init__(self, location: tuple[int, int]):
        super().__init__(Tile.STATIC, location)

    def select_action(self, state: np.ndarray, agent_location: tuple[int, int], is_scattering: bool) -> int:
        return Action.NONE.value
