import numpy as np
from tiles.tile import Tile

class Tiles(np.ndarray):

    def __new__(cls, array):
        obj = np.array(array, dtype=Tile).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.extra_info = getattr(obj, 'extra_info', None)

    def to_integer_array(self) -> np.ndarray[int]:
        return np.vectorize(lambda tile: tile.id)(self)

