from sklearn.base import BaseEstimator
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

    def fit(self, data: pd.DataFrame) -> None:
        assert self._input_col in data
        self._fitted = True

        column_preprocessed = data[self._input_col].apply(self._str_preprocessor).to_list()

        self._transformer.fit(column_preprocessed)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self._fitted
        assert self._input_col in data

        preprocessed_col = self._transformer.transform(data[self._input_col].apply(self._str_preprocessor)).toarray()
        
        columns = [self._output_col_prefix + str(idx) for idx in range(len(self._transformer.vocabulary_))]
        preprocessed_data = pd.DataFrame(preprocessed_col, columns=columns)

        data = pd.concat([data, preprocessed_data], axis=1)
        data.drop(self._input_col, axis=1, inplace=True)

        return data
