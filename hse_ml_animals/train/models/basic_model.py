import numpy as np
import pandas as pd
import typing as T

from hse_ml_animals.train.models.model import BaseModel
from hse_ml_animals.train.models import base_model_from_config
import hse_ml_animals.utils as utils


class BasicModel(BaseModel):
    def __init__(
            self,
            base_model_config: T.Dict,
            test_data: pd.DataFrame = None,
            test_pseudo: pd.DataFrame = None,
            **params
        ):
        self._base_model_config = base_model_config
        
        if test_data and test_pseudo:
            self._pseudo_data = pd.merge(test_data, test_pseudo, left_index=True, right_index=True)
        else:
            assert not test_data and not test_pseudo
            self._pseudo_data = None

        self._model = base_model_from_config(self._base_model_config)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,) -> None:
        if self._pseudo_data:
            X_pseudo = self._pseudo_data.drop(utils.TARGET, axis=1)
            y_pseudo = self._pseudo_data[utils.TARGET]

            X = pd.concat([X_train, X_pseudo])
            y = pd.concat([y_train, y_pseudo])
        else:
            X, y = X_train, y_train

        self._model.fit(X, y)     

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        assert utils.TARGET not in data

        return self._model.predict(data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        assert utils.TARGET not in data

        return self._model.predict_proba(data)
