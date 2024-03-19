class House:
    def __init__(self, house_tiles: list[tuple[int, int]], exit_target: tuple[int, int]):
        self.house_locations = house_tiles
        self.exit_target = exit_target