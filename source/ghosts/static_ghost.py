import numpy as np
from ghosts import Ghost
from houses.house import House
from states.state import State
from tiles.tile import Tile
from actions.action import Action

class StaticGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        super().__init__(location, house)
        self.tile = Tile.STATIC
        self.scatter_target = location
        self.wait_time = 0

    def select_action(self, state: State) -> Action:
        return Action.NONE
