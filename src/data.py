import pandas as pd
from pathlib import Path
import py7zr


def extract_data(data_path: Path, out_path: Path):
    OUT_PATH.mkdir(exist_ok=True)
    
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
        with py7zr.SevenZipFile(DATA_PATH / file, mode="r") as z:
            z.extractall(path=OUT_PATH)
    
    print("Done extracting!")


def load_data(out_path: Path):
    
    dtypes = {
        "store_nbr": "int8",
        "item_nbr": "int32",
        "unit_sales": "float32"
    }

    train = pd.read_csv("/kaggle/working/favorita/train.csv", dtype=dtypes, parse_dates=["date"])
    test = pd.read_csv("/kaggle/working/favorita/test.csv", dtype=dtypes,parse_dates=["date"])
    
    stores = pd.read_csv("/kaggle/working/favorita/stores.csv", dtype=dtypes)
    items = pd.read_csv("/kaggle/working/favorita/items.csv", dtype=dtypes)
    oil = pd.read_csv("/kaggle/working/favorita/oil.csv", dtype=dtypes, parse_dates=["date"])
    holidays = pd.read_csv("/kaggle/working/favorita/holidays_events.csv", dtype=dtypes, parse_dates=["date"])
    transactions = pd.read_csv("/kaggle/working/favorita/transactions.csv", dtype=dtypes, parse_dates=["date"])
    print("Файлы загружены, форма train:", train.shape)
    return train, test, stores, items, oil, holidays, transactions
