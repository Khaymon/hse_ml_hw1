import pandas as pd

from .base_preprocessor import BasePreprocessor
from .tf_idf_preprocessor import TfIdfPreprocessor


class ColorTfIdfPreprocessor(TfIdfPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.COLOR, "Color_", self._preprocess_colors_str, **params)

    def _preprocess_colors_str(self, color_str: str) -> str:
        return ' '.join(color_str.split('/'))
