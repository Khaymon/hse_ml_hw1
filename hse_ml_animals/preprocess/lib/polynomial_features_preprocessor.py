import numpy as np
import pandas as pd
import typing as T

from sklearn.preprocessing import PolynomialFeatures

from .base_preprocessor import BasePreprocessor
import hse_ml_animals.utils as utils


class PolynomialFeaturesPreprocessor(BasePreprocessor):
    def __init__(self, without_columns: T.Tuple[str] = (), **params):
        self._without_columns = without_columns
        self._columns = None

        self._poly_features = PolynomialFeatures(**params)


    def fit(self, data: pd.DataFrame) -> None:
        train_data = data.drop(utils.TARGET, axis=1)
        self._columns = list(filter(lambda column: column not in self._without_columns, train_data.columns))

        self._poly_features.fit(train_data[self._columns])


    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        transformed_columns = self._poly_features.transform(data[self._columns])

        result_columns = ["Poly_" + str(idx) for idx in range(self._poly_features.n_output_features_)]
        transformed_data = pd.DataFrame(transformed_columns, columns=result_columns)

        data = data.drop(self._columns, axis=1)
        data = pd.concat([data, transformed_data], axis=1)

        return data
