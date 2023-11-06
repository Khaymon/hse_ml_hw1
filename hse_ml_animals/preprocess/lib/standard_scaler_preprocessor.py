import pandas as pd
from sklearn.preprocessing import StandardScaler
import typing as T

from .base_preprocessor import BasePreprocessor

import hse_ml_animals.utils as utils


class StandardScalerPreprocessor(BasePreprocessor):
    def __init__(self, columns: T.Optional[str | T.List[str]] = None, without_columns: T.Iterable[str] = ()) -> None:
        super().__init__()

        self._fitted = False

        self._without_columns = without_columns or ()
        self._columns = columns
        self._scaler = StandardScaler()

    
    def fit(self, data: pd.DataFrame) -> None:
        if self._columns is None:
            train_data = data.drop(utils.TARGET, axis=1, errors="ignore")
            self._columns = list(filter(lambda column: column not in self._without_columns, train_data.columns))

        self._scaler.fit(data[self._columns])
        self._fitted = True

    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted

        to_scale_data = data[self._columns]
        other_data = data.drop(self._columns, axis=1)

        scaled_data = pd.DataFrame(self._scaler.transform(to_scale_data), columns=self._columns)

        data = pd.concat([other_data, scaled_data], axis=1)
        return data