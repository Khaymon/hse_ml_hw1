import pandas as pd

from ..base.base_preprocessor import BasePreprocessor
from ..base.vectorizer_preprocessor import VectorizerPreprocessor


class ColorVectorizerPreprocessor(VectorizerPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.COLOR, "Color_", self._preprocess_colors_str, **params)

        self._output_col = "NumColors"

    def _preprocess_colors_str(self, color_str: str) -> str:
        return ' '.join(color_str.split('/'))
