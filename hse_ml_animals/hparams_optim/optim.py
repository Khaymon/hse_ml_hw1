import argparse
from copy import deepcopy
import json
import optuna
import pandas as pd
import typing as T

import hse_ml_animals.utils as utils
from hse_ml_animals.eval import strategy_from_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Hparams optimizer",
        "Optimizes model's hparams with optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model-config", type=str, required=True, help="Path to the model config")
    parser.add_argument("--tune-config", type=str, required=True, help="Path to the tune config")
    parser.add_argument("--eval-config", type=str, required=True, help="Path to the eval config")
    parser.add_argument("--train-data", type=str, required=True, help="Path to the train dataset")
    parser.add_argument("--target-function", type=str, required=True, help="Name of the objective metric")

    parser.add_argument("--minimize", action="store_true", default=False, help="Whether to minimize the metric")

    return parser.parse_args()


def objective(
        trial,
        eval_config: T.Dict,
        model_config: T.Dict,
        tune_config: T.Dict,
        train_data: pd.DataFrame,
        target_function: str,
    ) -> float:
    trial_model_config = deepcopy(model_config)
    for tune_param in tune_config["tune_params"]:
        param_name = tune_param["path"][-1]
        if tune_param["type"] == "float":
            left, right = tune_param["values"]
            utils.set_path(trial_model_config, tune_param["path"], trial.suggest_float(param_name, left, right))
        elif tune_param["type"] == "int":
            left, right = tune_param["values"]
            utils.set_path(trial_model_config, tune_param["path"], trial.suggest_int(param_name, left, right))
        elif tune_param["type"] == "categorical":
            utils.set_path(trial_model_config, tune_param["path"], trial.suggest_categorical(param_name, tune_param["values"]))
        else:
            raise NotImplementedError(f"Type {tune_param['type']} is not implemented now")
    
    eval_strategy = strategy_from_config(eval_config)
    eval_result = eval_strategy.eval(trial_model_config, train_data)

    return eval_result[target_function]


def main():
    args = _parse_args()

    objective_wrapper = lambda trial: objective(
        trial=trial,
        eval_config=utils.read_json(args.eval_config),
        model_config=utils.read_json(args.model_config),
        tune_config=utils.read_json(args.tune_config),
        train_data=utils.read_pandas(args.train_data),
        target_function=args.target_function,
    )

    if args.minimize:
        study = optuna.create_study(direction="minimize")
    else:
        study = optuna.create_study(direction="maximize")
    study.optimize(objective_wrapper, n_trials=100, show_progress_bar=True, n_jobs=4)

    print(json.dumps(study.best_params, indent=4))


if __name__ == "__main__":
    main()