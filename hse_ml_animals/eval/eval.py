import argparse
import json
import typing as T

from hse_ml_animals.eval.strategies import *
import hse_ml_animals.utils as utils


def strategy_from_config(eval_config: T.Dict[str, T.Any]) -> BaseEvalStrategy:
    if eval_config["name"] == "train_test_split":
        return TrainTestSplitEvalStrategy(
            target=utils.TARGET,
            **eval_config["params"],
        )
    elif eval_config["name"] == "kfold":
        return KFoldEvalStrategy(
            target=utils.TARGET,
            **eval_config["params"],
        )
    else:
        raise NotImplementedError(f"Strategy {eval_config['name']} is not implemented")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Evaluator",
        "Evaluate model with given strategy and metric",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--eval-config", type=str, required=True, help="Path to the eval config")
    parser.add_argument("--model-config", type=str, required=True, help="Path to the model config")
    parser.add_argument("--train", type=str, required=True, help="Path to the train dataset")

    parser.add_argument("--eval-result", type=str, required=False, help="Path to the eval result")

    return parser.parse_args()


def main():
    args = _parse_args()

    strategy = strategy_from_config(utils.read_json(args.eval_config))
    train_data = utils.read_pandas(args.train)

    eval_result = strategy.eval(utils.read_json(args.model_config), train_data)
    if args.eval_result:
        utils.write_json(eval_result, args.eval_result)
    
    utils.print_json(eval_result)


if __name__ == "__main__":
    main()
