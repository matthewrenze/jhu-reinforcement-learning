import pytest
from unittest.mock import patch, Mock
from experiments.results import Results

@pytest.fixture
def setup():
    results_row = {"test_key": "test_value"}
    results = Results()
    return results_row, results

@patch("os.path.exists")
@patch("pandas.read_csv")
def test_load(mock_read_csv, mock_exists, setup):
    _, results = setup
    mock_exists.return_value = True
    mock_read_csv.return_value = Mock("test_table")
    results.load("results.csv")
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once_with("../data/results/results.csv")
    assert results._table == mock_read_csv.return_value

def test_add(setup):
    results_row, results = setup
    results.add(results_row)
    assert not results._table.empty
    assert results._table["test_key"][0] == "test_value"

@patch("pandas.DataFrame.to_csv")
def test_save(mock_to_csv, setup):
    _, results = setup
    results.save("results.csv")
    mock_to_csv.assert_called_once_with("../data/results/results.csv", index=False)




