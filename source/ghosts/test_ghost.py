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
        super().__init__(location, house)
        self.tile = Tile.STATIC
        self.scatter_target = scatter_target
        self.wait_time = 0

    def select_action(self, state: State) -> Action:
        return Action.NONE
