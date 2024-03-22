import pandas as pd
import warnings
warnings.simplefilter("ignore", FutureWarning)

FOLDER_PATH = "../details"

class Details():

    def __init__(self):
        self.table = pd.DataFrame()

    def add(self, details_row: dict) -> None:
        details_row = pd.DataFrame.from_records([details_row])
        self.table = pd.concat([self.table, details_row], ignore_index=True)

    def save(self) -> None:
        agent_name = self.table["agent_name"][0]
        curriculum = "curriculum" if self.table["curriculum"][0] else "non-curriculum"
        alpha = self.table["alpha"][0]
        gamma = self.table["gamma"][0]
        epsilon = self.table["epsilon"][0]
        episode = self.table["episode"][0]
        file_name = f"{agent_name} - {curriculum} - {alpha} - {gamma} - {epsilon} - {episode}.csv"
        file_path = f"{FOLDER_PATH}/{file_name}"
        self.table.to_csv(file_path, index=False)