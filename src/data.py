import pandas as pd
from pathlib import Path
import py7zr


def extract_data(data_path: Path, out_path: Path):
    out_path.mkdir(exist_ok=True)

    files = [
        "train.csv.7z",
        "test.csv.7z",
        "stores.csv.7z",
        "items.csv.7z",
        "oil.csv.7z",
        "holidays_events.csv.7z",
        "transactions.csv.7z",
        "sample_submission.csv.7z"
    ]

    for file in files:
        with py7zr.SevenZipFile(data_path / file, mode="r") as z:
            z.extractall(path=out_path)


def load_data(out_path: Path):
    
    dtypes = {
        "store_nbr": "int8",
        "item_nbr": "int32",
        "unit_sales": "float32"
    }

    train = pd.read_csv(out_path / "train.csv", dtype=dtypes, parse_dates=["date"])
    test = pd.read_csv(out_path / "test.csv", dtype=dtypes, parse_dates=["date"])

    stores = pd.read_csv(out_path / "stores.csv")
    items = pd.read_csv(out_path / "items.csv")
    oil = pd.read_csv(out_path / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(out_path / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(out_path / "transactions.csv", parse_dates=["date"])

    return train, test, stores, items, oil, holidays, transactions