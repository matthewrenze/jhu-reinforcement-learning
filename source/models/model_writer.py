import numpy as np
from models.model import Model
from joblib import dump
FOLDER_PATH = "../data/models"

class ModelWriter:
    def write(self, file_name: str, model: Model):

        if model.__class__.__name__ == "QTable":
            file_path = f"{FOLDER_PATH}/{file_name}.csv"
            np.savetxt(file_path, model.table, delimiter=",")

        if model.__class__.__name__ == "FeatureWeights": 
            file_path = f"{FOLDER_PATH}/{file_name}.csv"
            np.savetxt(file_path, model.table, delimiter=",")

        if model.__class__.__name__ == "DeepQNetwork":
            file_path = f"{FOLDER_PATH}/{file_name}.joblib"
            dump(model.model, file_path)

        

