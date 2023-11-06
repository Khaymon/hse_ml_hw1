import pandas as pd
import typing as T

from .base_preprocessor import BasePreprocessor


class DropPreprocessor(BasePreprocessor):
    def __init__(self, drop_columns: T.Sequence[str]):
        self._drop_columns = drop_columns

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop(self._drop_columns, axis=1, errors="ignore", inplace=True)

        return data
