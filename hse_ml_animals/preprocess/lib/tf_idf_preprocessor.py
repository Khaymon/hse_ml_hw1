import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import typing as T

from .text_encoder_preprocessor import TextEncoderPreprocessor


class TfIdfPreprocessor(TextEncoderPreprocessor):
    def __init__(
            self,
            input_col: str,
            output_col_suffix: str,
            str_preprocessor: T.Callable[[str], str] = None,
            **params
        ) -> None:
        super().__init__(input_col, output_col_suffix, TfidfVectorizer, str_preprocessor, **params)
