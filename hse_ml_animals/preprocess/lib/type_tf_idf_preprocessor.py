from .base_preprocessor import BasePreprocessor
from .tf_idf_preprocessor import TfIdfPreprocessor


class TypeTfIdfPreprocessor(TfIdfPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.ANIMAL_TYPE, "Type_", **params)
