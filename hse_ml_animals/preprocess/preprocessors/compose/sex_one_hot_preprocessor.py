from ..base.base_preprocessor import BasePreprocessor
from ..base.one_hot_preprocessor import OneHotPreprocessor


class SexOneHotPreprocessor(OneHotPreprocessor):
    def __init__(self):
        super().__init__(BasePreprocessor.InputColumns.SEX, "Sex_")
