import os
import numpy as np
from models.model import Model

FOLDER_PATH = "../data/models"

class QTable(Model):

    def __init__(self, table: np.ndarray = None):
        self.table = table

