from unittest.mock import patch
import numpy as np
from models.q_table import QTable

def test_init_with_none():
    q_table = QTable()
    assert q_table.table is None

def test_init_with_table():
    table = np.zeros((3, 3))
    q_table = QTable(table)
    assert q_table.table is table

@patch("os.path.exists")
@patch("numpy.loadtxt")
def test_load(mock_loadtxt, mock_exists):
    q_table = QTable()
    table = np.zeros((3, 3))
    mock_exists.return_value = True
    mock_loadtxt.return_value = table
    q_table.load("test_agent")
    mock_loadtxt.assert_called_once_with("../models/test_agent.csv", delimiter=",")
    assert q_table.table is mock_loadtxt.return_value

@patch("numpy.savetxt")
def test_save(mock_savetxt):
    q_table = QTable()
    q_table.table = np.zeros((3, 3))
    q_table.save("test")
    mock_savetxt.assert_called_once_with("../models/test.csv", q_table.table, delimiter=",")