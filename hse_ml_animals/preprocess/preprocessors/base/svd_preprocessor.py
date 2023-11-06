from dataclasses import dataclass
from datetime import datetime
import typing as T
import pandas as pd

from sklearn.decomposition import TruncatedSVD

from .base_preprocessor import BasePreprocessor

import hse_ml_animals.utils as utils


class SVDPreprocessor(BasePreprocessor):
    def __init__(self, without_columns: T.Iterable[str] = (), **params) -> None:
        self._without_columns = without_columns
        self._columns = None

        self._svd = TruncatedSVD(**params)


    def fit(self, data: pd.DataFrame) -> None:
        train_data = data.drop(utils.TARGET, axis=1, errors="ignore")
        self._columns = list(filter(lambda column: column not in self._without_columns, train_data.columns))
        
        self._svd.fit(data[self._columns])


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_columns = self._svd.transform(data[self._columns])
        
        result_columns = ["SVD_" + str(idx) for idx in range(self._svd.n_components)]
        transformed_data = pd.DataFrame(transformed_columns, columns=result_columns)

        data = data.drop(self._columns, axis=1)
        data = pd.concat([data, transformed_data], axis=1)

        return data
