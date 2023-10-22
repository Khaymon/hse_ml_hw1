import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import typing as T

from .base_strategy import BaseEvalStrategy, Metric

from hse_ml_animals.train import train_model, predict


class KFoldEvalStrategy(BaseEvalStrategy):
    def __init__(
            self,
            target: str,
            **params,
        ):
        super().__init__(target, **params)

        self._splitter = KFold(**params["splitter_params"])

    def eval(self, model_config: T.Dict[str, T.Any], train_data: pd.DataFrame) -> T.Dict[str, T.Any]:
        eval_result = self.get_base_result()
        for metric, metric_param in zip(self._mertrics, self._metric_params):
            metric_results = {}
            for idx, (train_idx, val_idx) in enumerate(self._splitter.split(train_data)):
                train_train_data, train_val_data = train_data.iloc[train_idx], train_data.iloc[val_idx]

                model = train_model(model_config, train_train_data)

                X_val = train_val_data.drop(self._target, axis=1)
                y_val = train_val_data[self._target]
                predictions = predict(model, X_val)

                metric_results[f"split_{idx}"] = metric(y_val, predictions, **metric_param)
            
            eval_result[f"{metric.__name__}_avg"] = np.mean(list(metric_results.values()))
            eval_result[metric.__name__] = metric_results
        
        return eval_result
