from ..base.base_preprocessor import BasePreprocessor
from ..base.tf_idf_preprocessor import TfIdfPreprocessor


class TypeTfIdfPreprocessor(TfIdfPreprocessor):
    def __init__(self, **params):
        super().__init__(BasePreprocessor.InputColumns.ANIMAL_TYPE, "Type_", **params)
