import argparse
import pandas as pd
from pathlib import Path
import typing as T

import hse_ml_animals.preprocess.preprocessors as preprocess_lib
import hse_ml_animals.utils as utils


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Preprocessor",
        "Preprocess set with preprocessing lib",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config", type=Path, required=True, help="Path to the preprocessor config")
    parser.add_argument("--train-path", type=Path, required=True, help="Path to the train")
    parser.add_argument("--test-path", type=Path, required=True, help="Path to the test")
    parser.add_argument("--preprocessed-train-path", type=Path, required=True, help="Path to the preprocessed train")
    parser.add_argument("--preprocessed-test-path", type=Path, required=True, help="Path to the preprocessed test")

    return parser.parse_args()


def preprocessor_from_config(preprocessor_config: T.Dict) -> preprocess_lib.SequentialPreprocessor:
    preprocessors = []
    for config in preprocessor_config["preprocessors"]:
        preprocessors.append(getattr(preprocess_lib, config["name"])(**config.get("params", {})))

    return preprocess_lib.SequentialPreprocessor(preprocessors)


def main():
    args = _parse_args()

    preprocessor = preprocessor_from_config(utils.read_json(args.config))

    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)

    preprocessed_train = preprocessor.fit_transform(train)
    preprocessed_test = preprocessor.transform(test)

    preprocessed_train.to_csv(args.preprocessed_train_path, index=False)
    preprocessed_test.to_csv(args.preprocessed_test_path, index=False)


if __name__ == "__main__":
    main()
