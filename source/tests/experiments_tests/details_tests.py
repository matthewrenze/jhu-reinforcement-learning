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
        "episode": 4}
    details = Details()
    return details, details_row

def test_add(setup):
    details, details_row = setup
    details.add(details_row)
    assert details.table["agent_name"][0] == "test_agent"

@patch("pandas.DataFrame.to_csv")
def test_save(mock_to_csv, setup):
    details, details_row = setup
    details.add(details_row)
    details.save("../folder/file.csv")
    mock_to_csv.assert_called_once_with("../folder/file.csv", index=False)


