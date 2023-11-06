from .base_preprocessor import BasePreprocessor
from .one_hot_preprocessor import OneHotPreprocessor


class TypeOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self):
        super().__init__(BasePreprocessor.InputColumns.ANIMAL_TYPE, "Type_")
