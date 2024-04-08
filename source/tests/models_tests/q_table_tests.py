import numpy as np
from models.q_table import QTable

def test_init_with_none():
    q_table = QTable()
    assert q_table.table is None

def test_init_with_table():
    table = np.zeros((3, 3))
    q_table = QTable(table)
    assert q_table.table is table