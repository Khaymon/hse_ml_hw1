from dataclasses import dataclass
from datetime import datetime
import typing as T
import pandas as pd

from ..base.base_preprocessor import BasePreprocessor


class DateTimePreprocessor(BasePreprocessor):
    @dataclass
    class OutputColumns:
        DAY = "day"
        MONTH = "month"
        WEEKDAY = "weekday"
        YEAR = "year"
        HOUR = "hour"
        MINUTE = "minute"

    def __init__(self) -> None:
        self._input_col = BasePreprocessor.InputColumns.DATE_TIME

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        assert self._input_col in data
        dates = data[self._input_col].apply(datetime.fromisoformat)

        data[DateTimePreprocessor.OutputColumns.DAY] = dates.dt.day
        data[DateTimePreprocessor.OutputColumns.MONTH] = dates.dt.month
        data[DateTimePreprocessor.OutputColumns.WEEKDAY] = dates.dt.weekday
        data[DateTimePreprocessor.OutputColumns.YEAR] = dates.dt.year
        data[DateTimePreprocessor.OutputColumns.HOUR] = dates.dt.hour
        data[DateTimePreprocessor.OutputColumns.MINUTE] = dates.dt.minute

        data.drop(self._input_col, axis=1, inplace=True)

        return data
