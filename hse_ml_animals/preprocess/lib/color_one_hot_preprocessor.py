from .base_preprocessor import BasePreprocessor
from .one_hot_preprocessor import OneHotPreprocessor


class ColorOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.COLOR, "Color_", **params)
