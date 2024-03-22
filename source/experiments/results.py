import os.path

import pandas as pd

FILE_PATH = "../results/results.csv"

class Results():

    def __init__(self) -> None:
        self._table = pd.DataFrame()
        self._create()

    def load(self) -> None:
        if os.path.exists(FILE_PATH):
            self._table = pd.read_csv(FILE_PATH)

    def add(self, results_row: dict) -> None:
        results_row = pd.DataFrame.from_records([results_row])
        self._table = pd.concat([self._table, results_row], ignore_index=True)

    def save(self) -> None:
        self._table.to_csv(FILE_PATH, index=False)

    def _create(self) -> None:
        self._table = pd.DataFrame()