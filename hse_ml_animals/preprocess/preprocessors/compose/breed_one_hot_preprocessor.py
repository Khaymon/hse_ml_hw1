from ..base.base_preprocessor import BasePreprocessor
from ..base.one_hot_preprocessor import OneHotPreprocessor


class BreedOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.BREED, "Breed_", **params)
