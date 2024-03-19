from houses.house import House

class HouseFactory():

    # TODO: This needs to be dynamic created based on the level map
    # NOTE: Maybe I should use an "x" to indicate the house exit location
    # NOTE: ghosts always go left when leaving the house
    def create(self) -> House:
        house_locations = [(8, 10), (9, 10), (9, 9), (9, 11)]
        house_exit_location = (7, 9)
        return House(house_locations, house_exit_location)