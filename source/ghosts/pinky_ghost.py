from ghosts.ghost import Ghost
from houses.house import House
from tiles.tile import Tile
from actions.action import Action

class PinkyGhost(Ghost):

    def __init__(self, location: tuple[int, int], house: House):
        super().__init__(location, house)
        self.tile = Tile.PINKY
        self.scatter_target = (0, 4)
        self.wait_time = 4

    def _get_chase_target(
            self,
            agent_location: tuple[int, int],
            agent_orientation: int,
            ghost_locations: [list[tuple[int, tuple[int, int]]]]) -> tuple[int, int]:
        agent_orientation = Action(agent_orientation)
        if agent_orientation == Action.UP:
            return agent_location[0] - 4, agent_location[1]
        if agent_orientation == Action.DOWN:
            return agent_location[0] + 4, agent_location[1]
        if agent_orientation == Action.LEFT:
            return agent_location[0], agent_location[1] - 4
        if agent_orientation == Action.RIGHT:
            return agent_location[0], agent_location[1] + 4
        if agent_orientation == Action.NONE:
            return agent_location
        else:
            raise ValueError("Invalid orientation")


