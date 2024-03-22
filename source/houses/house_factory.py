from houses.house import House

class HouseFactory():

    # TODO: This uses a simple hack to keep ghosts from respawning on curriculum levels 1-9
    # TODO: If there is no house locations, then ghosts don't respawn
    def create(self, level) -> House:

        if level != 10:
            return House([], (0, 0))

        house_locations = [(8, 10), (9, 10), (9, 9), (9, 11)]
        house_exit_location = (7, 9)
        return House(house_locations, house_exit_location)