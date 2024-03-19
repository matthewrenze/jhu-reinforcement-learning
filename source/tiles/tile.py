from enum import Enum

class Tile(Enum):
    EMPTY = (0, ' ', 0)
    WALL = (1, '#', 0)
    PACMAN = (2, 'c', 0)
    DOT = (3, '.', 10)
    POWER = (4, 'o', 50)
    STATIC = (5, 's', 200)
    BLINKY = (6, 'b', 200)
    PINKY = (7, 'p', 200)
    INKY = (8, 'i', 200)
    CLYDE = (9, 'y', 200)

    def __init__(self, id: int, symbol: str, reward: int):
        self.id = id
        self.symbol = symbol
        self.reward = reward

    @classmethod
    def get_enum_from_id(cls, id: int):
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
    def get_symbol_from_id(cls, id: int) -> str:
        for tile in cls:
            if tile.id == id:
                return tile.symbol
        raise ValueError(f"No tile with id {id} found")

    @classmethod
    def get_id_from_symbol(cls, symbol: str) -> int:
        for tile in cls:
            if tile.symbol == symbol:
                return tile.id
        raise ValueError(f"No tile with symbol '{symbol}' found")

    def __str__(self):
        return str(self.name)

    def __repr__(self):
        return str(self.name)