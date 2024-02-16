from typing import Tuple
from enum import Enum
import numpy as np

class Tile(Enum):
    EMPTY = (0, ' ', 0)
    WALL = (1, '#', 0)
    PACMAN = (2, 'c', 0)
    GHOST = (3, 'm', 200)
    DOT = (4, '.', 10)
    POWER = (5, 'o', 50)
    BONUS = (6, '$', 100)

    # Note: For ghosts should we have s:static, r:random, b:blinky, p:pinky, i:inky, y:clyde?
    # Note: I think one of the original ghosts might already be random, so we just need one of the two

    def __init__(self, id: int, symbol:str, reward:int):
        self.id = id
        self.symbol = symbol
        self.reward = reward

    @classmethod
    def get_enum_from_id(cls, id:int):
        for tile in cls:
            if tile.id == id:
                return tile
        raise ValueError(f"No tile with id {id} found")

    @classmethod
    def get_enum_from_symbol(cls, symbol:str):
        for tile in cls:
            if tile.symbol == symbol:
                return tile
        raise ValueError(f"No tile with symbol '{symbol}' found")

    @classmethod
    def get_symbol_from_id(cls, id:int) -> str:
        for tile in cls:
            if tile.id == id:
                return tile.symbol
        raise ValueError(f"No tile with id {id} found")

    @classmethod
    def get_id_from_symbol(cls, symbol:str) -> int:
        for tile in cls:
            if tile.symbol == symbol:
                return tile.id
        raise ValueError(f"No tile with symbol '{symbol}' found")

def get_agent_location(state:np.ndarray) -> Tuple[int, int]:
    tile_id = Tile.get_id_from_symbol(Tile.PACMAN.symbol)
    location = np.where(state == tile_id)
    if len(location[0]) == 0:
        raise ValueError(f"No tile with agent found in state")
    if len(location[0]) > 1:
        raise ValueError(f"Multiple tiles with agent found in state")
    row = int(location[0][0])
    col = int(location[1][0])
    return row, col

def get_ghost_locations(state:np.ndarray) -> list[Tuple[int, int]]:
    tile_id = Tile.get_id_from_symbol(Tile.GHOST.symbol)
    locations = np.where(state == tile_id)
    return list(zip(locations[0], locations[1]))
