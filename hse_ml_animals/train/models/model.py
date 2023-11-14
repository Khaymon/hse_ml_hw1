from copy import deepcopy
import numpy as np
import pandas as pd
import typing as T

import hse_ml_animals.train.models.sklearn_models as sklearn_models
from hse_ml_animals.train.models.sklearn_models import *

class BaseModel(ClassifierMixin):
    def fit(self, data: pd.DataFrame, target: pd.Series) -> None:
        raise NotImplementedError()
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError()


def base_model_from_config(model_config: T.Dict[str, T.Any]) -> ClassifierMixin:
    if model_config["name"] == "RandomForestClassifier":
        return RandomForestClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "DecisionTreeClassifier":
        return DecisionTreeClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "LogisticRegression":
        return LogisticRegression(**model_config.get("params", {}))
    elif model_config["name"] == "GradientBoostingClassifier":
        return GradientBoostingClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "KNeighborsClassifier":
        return KNeighborsClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "ExtraTreesClassifier":
        return ExtraTreesClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "MLPClassifier":
        return MLPClassifier(**model_config.get("params", {}))
    elif model_config["name"] == "StackingClassifier":
        estimators = []
        for estimator in model_config["params"]["estimators"]:
            model_name = estimator["name"]
            model = getattr(sklearn_models, estimator["model"]["name"])(**estimator["model"].get("params", {}))
            estimators.append((model_name, model))

        final_estimator_config = model_config["params"]["final_estimator"]
        final_estimator = getattr(sklearn_models, final_estimator_config["name"])(**final_estimator_config.get("params", {}))

        other_params = deepcopy(model_config["params"])


        other_params.pop("estimators", None)
        other_params.pop("final_estimator", None)

        return StackingClassifier(estimators, final_estimator, **other_params)
    elif model_config["name"] == "SVC":
        return SVC(**model_config.get("params", {}))
    else:
        raise NotImplementedError(f"Model {model_config['name']} is not implemented")
