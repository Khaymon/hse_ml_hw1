from dataclasses import dataclass
import pandas as pd

from ..base.base_preprocessor import BasePreprocessor


class AgePreprocessor(BasePreprocessor):
    @dataclass
    class OutputColumns:
        DAYS = "DaysUponOutcome"

    def __init__(self) -> None:
        self._input_col = BasePreprocessor.InputColumns.AGE
        self._output_col = AgePreprocessor.OutputColumns.DAYS

    def fit(self, data: pd.DataFrame) -> None:
        return
    
    @staticmethod
    def _age_to_days(age_str: str) -> int:
        if not age_str:
            return 0

        amount, interval = age_str.split()

        if "month" in interval:
            return int(amount) * 30
        elif "year" in interval:
            return int(amount) * 365
        elif "week" in interval:
            return int(amount) * 7
        elif "day" in interval:
            return int(amount)
        
        raise ValueError(f"Interval {interval} is unknown")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self._input_col not in data:
            raise ValueError(f"Preprocessor {type(self).__name__} needs column {BasePreprocessor.InputColumns.AGE}")
        
        data[self._input_col] = data[self._input_col].fillna("")
        data[self._output_col] = data[self._input_col].apply(self._age_to_days)
        data.drop(self._input_col, axis=1, inplace=True)

        return data
