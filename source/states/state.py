import numpy as np

# NOTE: Use this if I need it; otherwise, delete it
class State():
    def __init__(
            self,
            tiles: np.ndarray[int],
            agent_location: tuple[int, int],
            ghost_locations: list[tuple[int, int]],
            is_invincible: bool,
            ghost_mode: int):
        self.tiles = tiles
        self.agent_location = agent_location
        self.ghost_locations = ghost_locations
        self.is_invincible = is_invincible
        self.ghost_mode = ghost_mode
