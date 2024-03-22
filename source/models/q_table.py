import os
import numpy as np
from models.model import Model

FOLDER_PATH = "../models"

class QTable(Model):

    def __init__(self, table: np.ndarray = None):
        self.table = table

    # TODO: This is just temporary code to allow you to load and save trained models
    # TODO: It probably doesn't make sense to have load and save methods in the subclass
    def load(self, agent_name: str):
        file_name = f"{FOLDER_PATH}/{agent_name}.csv"
        if os.path.exists(file_name):
            self.table = np.loadtxt(file_name, delimiter=",")

    def save(self, agent_name: str) -> None:
        file_name = f"{FOLDER_PATH}/{agent_name}.csv"
        np.savetxt(file_name, self.table, delimiter=",")

