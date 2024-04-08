import os.path

import pandas as pd

FOLDER_PATH = "../data/results"

class Results():

    def __init__(self) -> None:
        self._table = pd.DataFrame()

    def load(self, file_name) -> None:
        if os.path.exists(FOLDER_PATH + "/" + file_name):
            self._table = pd.read_csv(FOLDER_PATH + "/" + file_name)

    def add(self, results_row: dict) -> None:
        results_row = pd.DataFrame.from_records([results_row])
        self._table = pd.concat([self._table, results_row], ignore_index=True)

    def save(self, file_name) -> None:
        self._table.to_csv(FOLDER_PATH + "/" + file_name, index=False)