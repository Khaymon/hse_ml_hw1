from .base_preprocessor import BasePreprocessor
from .one_hot_preprocessor import OneHotPreprocessor


class BreedOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.BREED, "Breed_", **params)
