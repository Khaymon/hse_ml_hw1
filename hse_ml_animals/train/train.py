import argparse
import numpy as np
import typing as T
import pandas as pd
from sklearn.base import ClassifierMixin

import hse_ml_animals.utils as utils

from hse_ml_animals.train.models.model import BaseModel
from hse_ml_animals.train.models.basic_model import BasicModel
from hse_ml_animals.train.models.kfold_blending_model import KFoldBlendingModel

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Trainer",
        "Train model from config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-config", type=str, required=True, help="Path to the model config")
    parser.add_argument("--train", type=str, required=True, help="Path to the train dataset")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset")

    parser.add_argument("--out-model", type=str, required=False, help="Path to the trained model")
    parser.add_argument("--out-predictions", type=str, required=False, help="Path to the test predictions")

    return parser.parse_args()


def compose_model_from_config(model_config: T.Dict) -> BaseModel:
    if model_config["name"] == "BasicModel":
        return BasicModel(model_config["base_model_config"], **model_config.get("params", {}))
    elif model_config["name"] == "KFoldBlending":
        return KFoldBlendingModel(model_config["base_model_config"], **model_config.get("params", {}))
    else:
        raise NotImplementedError(f"Compose model {model_config['name']} is not implemented")


def train_model(model_config: T.Dict[str, T.Any], train_data: pd.DataFrame) -> ClassifierMixin:
    assert utils.INDEX_COL not in train_data

    model = compose_model_from_config(model_config)

    X_train = train_data.drop(utils.TARGET, axis=1)
    y_train = train_data[utils.TARGET]

    model.fit(X_train, y_train)
    return model


def predict(model: ClassifierMixin, test_data: pd.DataFrame) -> np.ndarray:
    assert utils.INDEX_COL not in test_data
    assert utils.TARGET not in test_data

    return model.predict(test_data)


def main():
    args = _parse_args()

    train_data = pd.read_csv(args.train, index_col=utils.INDEX_COL)
    test_data = pd.read_csv(args.test, index_col=utils.INDEX_COL)

    model = train_model(utils.read_json(args.model_config), train_data)

    if args.out_model:
        utils.pickle_dump(model, args.out_model)
    
    if args.out_predictions:
        predictions = pd.DataFrame(predict(model, test_data), index=test_data.index, columns=[utils.TARGET])
        predictions.to_csv(args.out_predictions)


if __name__ == "__main__":
    main()