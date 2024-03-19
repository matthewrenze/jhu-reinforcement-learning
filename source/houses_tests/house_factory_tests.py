from houses.house_factory import HouseFactory
from actions.action import Action

def test_create():
    expected_locations = [(8, 10), (9, 10), (9, 9), (9, 11)]
    expected_exit_location = (7, 9)
    house_factory = HouseFactory()
    actual_house = house_factory.create()
    assert actual_house.house_locations == expected_locations
    assert actual_house.exit_target == expected_exit_location
