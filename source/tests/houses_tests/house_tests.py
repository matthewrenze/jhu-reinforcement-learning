from houses.house import House

def test_init():
    house_tiles = [(7, 10), (8, 10), (9, 10), (9, 9), (9, 11)]
    exit_target = (7, 9)
    house = House(house_tiles, exit_target)
    assert house.house_locations == house_tiles
    assert house.exit_target == exit_target