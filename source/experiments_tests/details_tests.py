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
    columns = details.table.columns
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
    assert len(details.table) == 1
    assert details.table["agent_name"][0] == "test_agent"
    assert details.table["curriculum"][0]
    assert details.table["alpha"][0] == 0.1
    assert details.table["gamma"][0] == 0.2
    assert details.table["epsilon"][0] == 0.3
    assert details.table["mode"][0] == "train"
    assert details.table["episode"][0] == 4
    assert details.table["game_level"][0] == 5
    assert details.table["time_step"][0] == 6
    assert details.table["reward"][0] == 7
    assert details.table["total_reward"][0] == 8

@patch("pandas.DataFrame.to_csv")
def test_save(mock_to_csv, setup):
    details, details_row = setup
    details.add(details_row)
    details.save()
    mock_to_csv.assert_called_once_with("../details/test_agent - curriculum - 0.1 - 0.2 - 0.3 - 4.csv", index=False)


