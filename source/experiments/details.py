import pandas as pd
import warnings
warnings.simplefilter("ignore", FutureWarning)

class Details():

    def __init__(self):
        self.table = pd.DataFrame()

    def add(self, details_row: dict) -> None:
        details_row = pd.DataFrame.from_records([details_row])
        self.table = pd.concat([self.table, details_row], ignore_index=True)

    def save(self, file_path) -> None:
        self.table.to_csv(file_path, index=False)