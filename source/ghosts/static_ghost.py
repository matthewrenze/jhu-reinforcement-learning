import numpy as np
from ghosts import Ghost
from houses.house import House
from states.state import State
from tiles.tile import Tile
from actions.action import Action

class StaticGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        tile = Tile.STATIC
        scatter_target = location
        super().__init__(tile, location, scatter_target, house)

    def select_action(self, state: State) -> Action:
        return Action.NONE
