import pandas as pd
from sklearn.cluster import KMeans
import typing as T

from .base_preprocessor import BasePreprocessor
import hse_ml_animals.utils as utils


class KMeansPreprocessor(BasePreprocessor):
    def __init__(self, without_columns: T.List[str], **params):
        self._without_columns = without_columns
        self._columns = None
        self._out_column = "Cluster"

        self._kmeans = KMeans(**params)

    def fit(self, data: pd.DataFrame) -> None:
        train_data = data.drop(utils.TARGET, axis=1)
        self._columns = list(filter(lambda column: column not in self._without_columns, train_data.columns))

        self._kmeans.fit(data[self._columns])
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._out_column] = self._kmeans.predict(data[self._columns])

        return data
