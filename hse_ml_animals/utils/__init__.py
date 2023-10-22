import pandas as pd
from pathlib import Path
import pickle
import typing as T
import json

INDEX_COL = "ID"
TARGET = "Outcome"


def read_pandas(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=INDEX_COL)


def read_json(path: Path) -> T.Dict[str, T.Any]:
    with open(path, 'r') as input_file:
        return json.load(input_file)


def write_json(value: T.Dict[str, T.Any], path: Path) -> None:
    with open(path, 'w') as output_file:
        json.dump(value, output_file, indent=4)


def pickle_dump(value: T.Any, path: Path) -> None:
    with open(path, 'wb') as output_file:
        pickle.dump(value, output_file)


def pickle_load(path: Path) -> T.Any:
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def set_path(values: T.Dict, path: T.List[str], new_value: T.Any) -> None:
    current_values = values
    for idx, key in enumerate(path):
        if idx == len(path) - 1:
            current_values[key] = new_value
        else:
            if key not in current_values:
                current_values[key] = {}
            current_values = current_values[key]


def print_json(values: T.Dict) -> None:
    print(json.dumps(values, indent=4))
