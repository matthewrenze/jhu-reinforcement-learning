import numpy as np
from ghosts.ghost import Ghost
from tiles.tile import Tile
from agents.actions import Action

class TestGhost(Ghost):

    def __init__(self, location: tuple[int, int] = (0, 0)):
        super().__init__(Tile.STATIC, location)

    def select_action(self, state: np.ndarray, agent_location: tuple[int, int], is_scattering: bool) -> int:
        return Action.NONE.value
