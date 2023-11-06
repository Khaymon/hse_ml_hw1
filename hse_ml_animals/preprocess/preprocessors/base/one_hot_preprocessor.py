from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import typing as T

from .text_encoder_preprocessor import TextEncoderPreprocessor


class OneHotPreprocessor(TextEncoderPreprocessor):
    def __init__(
            self,
            input_col: str,
            output_col_suffix: str,
            str_preprocessor: T.Callable[[str], str] = None,
            **params
        ):
        super().__init__(input_col, output_col_suffix, OneHotEncoder, str_preprocessor, **params)
