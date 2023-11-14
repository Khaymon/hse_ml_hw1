from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd
import typing as T

from .base_preprocessor import BasePreprocessor


class TextEncoderPreprocessor(BasePreprocessor):
    def __init__(
            self,
            input_col: str,
            output_col_prefix: str,
            transformer: BaseEstimator,
            str_preprocessor: T.Callable[[str], str] = None,
            **params
        ):
        self._input_col = input_col
        self._output_col_prefix = output_col_prefix

        self._fitted = False
        self._transformer = transformer(**params)
        if str_preprocessor:
            self._str_preprocessor = str_preprocessor
        else:
            self._str_preprocessor = lambda x: x
            
    def _to_array(self, column: pd.Series) -> np.ndarray:
        return column.to_numpy().reshape((-1, 1))
    
    def _to_list(self, column: pd.Series) -> T.List[T.Any]:
        return column.to_list()

    def fit(self, data: pd.DataFrame) -> None:
        assert self._input_col in data
        self._fitted = True

        try:
            column_preprocessed = data[self._input_col].apply(self._str_preprocessor).to_list()
            self._transformer.fit(column_preprocessed)
        except ValueError:
            column_preprocessed = data[self._input_col].apply(self._str_preprocessor).to_numpy().reshape((-1, 1))
            self._transformer.fit(column_preprocessed)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted
        assert self._input_col in data
        
        
        preprocessed_column = data[self._input_col].apply(self._str_preprocessor)
        try:
            result_data = self._transformer.transform(self._to_list(preprocessed_column)).toarray()
        except ValueError:
            result_data = self._transformer.transform(self._to_array(preprocessed_column)).toarray()
        
        columns = [self._output_col_prefix + str(idx) for idx in range(result_data.shape[1])]
        preprocessed_data = pd.DataFrame(result_data, columns=columns)

        data = pd.concat([data, preprocessed_data], axis=1)

        return data
