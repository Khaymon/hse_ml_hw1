import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import typing as T

Metric = T.Callable[[np.ndarray, np.ndarray], float]


def metric_from_str(metric_name: str) -> Metric:
    if metric_name == "accuracy":
        return accuracy_score
    elif metric_name == "f1":
        return f1_score
    else:
        raise NotImplementedError(f"Metric {metric_name} is not implemented")


class BaseEvalStrategy:
    def __init__(self, target: str, metric: Metric | T.List[Metric], metric_param: T.Dict | T.List[T.Dict], **params):
        self._target = target

        if not isinstance(metric, list):
            self._mertrics = [metric]
        else:
            self._mertrics = metric

        self._mertrics = [metric_from_str(metric) for metric in self._mertrics]
        
        if not isinstance(metric_param, list):
            self._metric_params = [metric_param]
        else:
            self._metric_params = metric_param

        self._params = params

        assert len(self._mertrics) == len(self._metric_params)

    def get_base_result(self) -> T.Dict:
        return {
            "strategy": type(self).__name__,
            **self._params,
        }

    def eval(self, model_config: T.Dict[str, T.Any], train_data: pd.DataFrame) -> T.Dict[str, T.Any]:
        raise NotImplementedError()
