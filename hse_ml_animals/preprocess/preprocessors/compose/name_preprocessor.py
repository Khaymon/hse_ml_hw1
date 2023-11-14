import numpy as np
import pandas as pd

from ..base.base_preprocessor import BasePreprocessor


class NamePreprocessor(BasePreprocessor):
    def __init__(self):
        self._input_col = "Name"
        self._output_col = "NameLength"

    def _name_length(self, name: str) -> int:
        if name is np.nan:
            return 0
        return len(name)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self._output_col] = data[self._input_col].apply(self._name_length)

        return data
