import typing as T

from ..base.base_preprocessor import BasePreprocessor
from ..base.tf_idf_preprocessor import TfIdfPreprocessor


class BreedTfIdfPreprocessor(TfIdfPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.BREED, "Breed_", self._preprocess_colors_str, **params)

    def _preprocess_colors_str(self, color_str: str) -> str:
        return ' '.join(color_str.split('/'))
