import typing as T

from .base_preprocessor import BasePreprocessor
from .vectorizer_preprocessor import VectorizerPreprocessor


class BreedVectorizerPreprocessor(VectorizerPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.BREED, "Breed_", self._preprocess_colors_str, **params)

    def _preprocess_colors_str(self, color_str: str) -> str:
        return ' '.join(color_str.split('/'))
