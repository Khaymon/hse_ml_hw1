import argparse
import numpy as np
import pandas as pd

import hse_ml_animals.utils as utils


def _parse_args():
    parser = argparse.ArgumentParser(
        "Pseudo labels",
        "Produce pseudo labels with from given model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--model", type=str, required=True, help="Path to the model pickle file")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--result", type=str, required=True, help="Path to the pseudo labels")
    
    parser.add_argument("--probas-path", type=str, required=False, help="Path to the predicted classes probas")
    parser.add_argument("--threshold", type=float, default=0.95, required=False, help="Selection threshold")

    return parser.parse_args()


def main():
    args = _parse_args()

    model = utils.pickle_load(args.model)
    test_data = utils.read_pandas(args.test)

    max_probas = np.max(model.predict_proba(test_data), axis=1)
    predictions = model.predict(test_data)

    pseudo_data = pd.DataFrame({"Outcome": predictions, "Proba": max_probas}, index=test_data.index)
    pseudo_data["Passed"] = pseudo_data["Proba"] > args.threshold
    if args.probas_path:
        pseudo_data.to_csv(args.probas_path)
    
    result_data = pseudo_data[pseudo_data["Passed"]]["Outcome"]
    result_data.to_csv(args.result)


if __name__ == "__main__":
    main()