import pytest
from unittest.mock import patch, Mock
from experiments.results import Results

@pytest.fixture
def setup():
    results_row = {
        "agent_name": "test_agent",
        "curriculum": "True",
        "alpha": 0.1,
        "gamma": 0.2,
        "epsilon": 0.3,
        "episode": 4,
        "game_level": 5,
        "avg_reward": 6,
        "total_reward": 7}
    results = Results()
    return results_row, results

def test_init(setup):
    _, results = setup
    assert results._table.empty
    columns = results._table.columns
    assert len(columns) == 9
    assert columns[0] == "agent_name"
    assert columns[1] == "curriculum"
    assert columns[2] == "alpha"
    assert columns[3] == "gamma"
    assert columns[4] == "epsilon"
    assert columns[5] == "episode"
    assert columns[6] == "game_level"
    assert columns[7] == "total_time"
    assert columns[8] == "avg_reward"
    assert columns[9] == "total_reward"

@patch("os.path.exists")
@patch("pandas.read_csv")
def test_load(mock_read_csv, mock_exists, setup):
    _, results = setup
    mock_exists.return_value = True
    mock_read_csv.return_value = Mock("test_table")
    results.load()
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once_with("../results/results.csv")
    assert results._table == mock_read_csv.return_value

def test_add(setup):
    results_row, results = setup
    results.add(results_row)
    assert not results._table.empty
    assert results._table.shape == (1, 10)
    for key, value in results_row.items():
        assert results._table[key][0] == value

@patch("pandas.DataFrame.to_csv")
def test_save(mock_to_csv, setup):
    _, results = setup
    results.save()
    mock_to_csv.assert_called_once_with("../results/results.csv", index=False)




