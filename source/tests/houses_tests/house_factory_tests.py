from houses.house_factory import HouseFactory
from actions.action import Action


def test_create_levels_1_to_9():
    expected_locations = []
    expected_exit_location = (0, 0)
    house_factory = HouseFactory()
    for level in range(1, 10):
        actual_house = house_factory.create(level)
        assert actual_house.house_locations == expected_locations
        assert actual_house.exit_target == expected_exit_location

def test_create_level_10():
    expected_locations = [(8, 10), (9, 10), (9, 9), (9, 11)]
    expected_exit_location = (7, 9)
    house_factory = HouseFactory()
    actual_house = house_factory.create(10)
    assert actual_house.house_locations == expected_locations
    assert actual_house.exit_target == expected_exit_location
