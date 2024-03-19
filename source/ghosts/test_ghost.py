from ghosts.ghost import Ghost
from houses.house import House
from states.state import State
from tiles.tile import Tile
from actions.action import Action


class TestGhost(Ghost):

    def __init__(
            self,
            location: tuple[int, int] = (0, 0),
            house: House = House([(0, 0)], (0, 0)),
            scatter_target: tuple[int, int] = (0, 0)):
        super().__init__(Tile.STATIC, location, scatter_target, house)

    def select_action(self, state: State) -> Action:
        return Action.NONE
