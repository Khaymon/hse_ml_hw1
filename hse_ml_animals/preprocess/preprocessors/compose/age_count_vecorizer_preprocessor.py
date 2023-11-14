import numpy as np
import typing as T

from ..base.base_preprocessor import BasePreprocessor
from ..base.vectorizer_preprocessor import VectorizerPreprocessor


class AgeVectorizerPreprocessor(VectorizerPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.AGE, "Age_", self._preprocess_age, **params)
        
    def _preprocess_age(self, age: T.Optional[str]) -> str:
        return age if age is not np.nan else ''
