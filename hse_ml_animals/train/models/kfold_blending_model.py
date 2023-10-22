import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import typing as T

from hse_ml_animals.train.models.model import BaseModel
from hse_ml_animals.train.models import base_model_from_config
import hse_ml_animals.utils as utils


class KFoldBlendingModel(BaseModel):
    def __init__(self, base_model_config: T.Dict, **params):
        self._base_model_config = base_model_config

        self._models = []
        self._splitter = KFold(**params["splitter_params"])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        for train_idx, _ in self._splitter.split(X, y):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            model = base_model_from_config(self._base_model_config)
            model.fit(X_train, y_train)

            self._models.append(model)        


    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert utils.TARGET not in data
        assert utils.INDEX_COL not in data

        predictions = []
        for model in self._models:
            predictions.append(model.predict_proba(data))
        
        final_proba = np.stack(predictions, axis=0).mean(axis=0)
        return np.argmax(final_proba, axis=1)
