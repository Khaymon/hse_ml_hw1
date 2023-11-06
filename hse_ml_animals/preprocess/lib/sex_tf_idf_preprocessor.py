import numpy as np

from .base_preprocessor import BasePreprocessor
from .tf_idf_preprocessor import TfIdfPreprocessor


class SexTfIdfPreprocessor(TfIdfPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.SEX, "Sex_", self._text_preprocessor, **params)

    def _text_preprocessor(self, text: str) -> str:
        if text is np.nan:
            return ""
        return text
