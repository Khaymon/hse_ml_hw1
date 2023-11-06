from dataclasses import dataclass
import numpy as np
import pandas as pd
import typing as T


class BasePreprocessor:
    @dataclass
    class InputColumns:
        ID = "ID"
        NAME = "NAME"
        DATE_TIME = "DateTime"
        ANIMAL_TYPE = "AnimalType"
        SEX = "SexuponOutcome"
        AGE = "AgeuponOutcome"
        BREED = "Breed"
        COLOR = "Color"

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        self.fit(data)
        return self.transform(data)
