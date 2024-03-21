import pytest
from unittest.mock import patch
from experiments.details import Details

@pytest.fixture
def setup():
    details_row = {
        "agent_name": "test_agent",
        "curriculum": True,
        "alpha": 0.1,
        "gamma": 0.2,
        "epsilon": 0.3,
        "mode": "train",
        "episode": 4,
        "game_level": 5,
        "time_step": 6,
        "reward": 7,
        "total_reward": 8}
    details = Details()
    return details, details_row

def test_init(setup):
    details, _ = setup
    columns = details._table.columns
    assert len(columns) == 11
    assert columns[0] == "agent_name"
    assert columns[1] == "curriculum"
    assert columns[2] == "alpha"
    assert columns[3] == "gamma"
    assert columns[4] == "epsilon"
    assert columns[5] == "mode"
    assert columns[6] == "episode"
    assert columns[7] == "game_level"
    assert columns[8] == "time_step"
    assert columns[9] == "reward"
    assert columns[10] == "total_reward"

def test_add(setup):
    details, details_row = setup
    details.add(details_row)
    assert len(details._table) == 1
    assert details._table["agent_name"][0] == "test_agent"
    assert details._table["curriculum"][0]
    assert details._table["alpha"][0] == 0.1
    assert details._table["gamma"][0] == 0.2
    assert details._table["epsilon"][0] == 0.3
    assert details._table["mode"][0] == "train"
    assert details._table["episode"][0] == 4
    assert details._table["game_level"][0] == 5
    assert details._table["time_step"][0] == 6
    assert details._table["reward"][0] == 7
    assert details._table["total_reward"][0] == 8

@patch("pandas.DataFrame.to_csv")
def test_save(mock_to_csv, setup):
    details, details_row = setup
    details.add(details_row)
    details.save()
    mock_to_csv.assert_called_once_with("../details/test_agent - curriculum - 0.1 - 0.2 - 0.3 - 4.csv", index=False)


