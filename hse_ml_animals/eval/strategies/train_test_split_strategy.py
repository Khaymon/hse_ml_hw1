import typing as T
import pandas as pd
from sklearn.model_selection import train_test_split

from .base_strategy import BaseEvalStrategy, Metric

from hse_ml_animals.train import train_model, predict

class TrainTestSplitEvalStrategy(BaseEvalStrategy):
    def __init__(
            self,
            target: str,
            **params,
        ):
        super().__init__(target, **params)

        self._splitter_params = params["splitter_params"]


    def eval(self, model_config: T.Dict[str, T.Any], train_data: pd.DataFrame) -> T.Dict[str, T.Any]:
        train_train_data, train_val_data = train_test_split(train_data, **self._splitter_params)

        model = train_model(model_config, train_train_data)

        X_val = train_val_data.drop(self._target, axis=1)
        y_val = train_val_data[self._target]
        predictions = predict(model, X_val)

        result = self.get_base_result()

        for metric, metric_param in zip(self._mertrics, self._metric_params):
            result[metric.__name__] = metric(y_val, predictions, **metric_param)

        return result
