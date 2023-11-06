from .base_preprocessor import BasePreprocessor
from .one_hot_preprocessor import OneHotPreprocessor


class SexOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self):
        super().__init__(BasePreprocessor.InputColumns.SEX, "Sex_")
