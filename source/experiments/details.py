import pandas as pd
import warnings
warnings.simplefilter("ignore", FutureWarning)

FOLDER_PATH = "../details"

class Details():

    def __init__(self):
        self._table = pd.DataFrame(
            columns=[
                "agent_name",
                "curriculum",
                "alpha",
                "gamma",
                "epsilon",
                "mode",
                "episode",
                "game_level",
                "time_step",
                "reward",
                "total_reward"])

    def add(self, details_row: dict) -> None:
        details_row = pd.DataFrame.from_records([details_row], columns=self._table.columns)
        self._table = pd.concat([self._table, details_row], ignore_index=True)

    def save(self) -> None:
        agent_name = self._table["agent_name"][0]
        curriculum = "curriculum" if self._table["curriculum"][0] else "non-curriculum"
        alpha = self._table["alpha"][0]
        gamma = self._table["gamma"][0]
        epsilon = self._table["epsilon"][0]
        episode = self._table["episode"][0]
        file_name = f"{agent_name} - {curriculum} - {alpha} - {gamma} - {epsilon} - {episode}.csv"
        file_path = f"{FOLDER_PATH}/{file_name}"
        self._table.to_csv(file_path, index=False)