import pandas as pd
import typing as T

from .base_preprocessor import BasePreprocessor


class SequentialPreprocessor(BasePreprocessor):
    def __init__(self, preprocessors: T.Sequence[BasePreprocessor]):
        self._preprocessors = preprocessors

    def fit(self, data: pd.DataFrame) -> None:
        for preprocessor in self._preprocessors:
            preprocessor.fit(data)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for preprocessor in self._preprocessors:
            data = preprocessor.transform(data)
        
        return data
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        for preprocessor in self._preprocessors:
            data = preprocessor.fit_transform(data)
        
        return data
