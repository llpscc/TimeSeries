import pandas as pd


def preprocess_train(train: pd.DataFrame) -> pd.DataFrame:
    
    train = train.copy()

    # фильтр по дате
    train["date"] = pd.to_datetime(train["date"])
    train = train[train["date"] >= "2016-08-16"]

    # onpromotion → int
    train["onpromotion"] = train["onpromotion"].astype(int)

    return train


def select_top_items(train: pd.DataFrame, top_n: int = 5000) -> pd.DataFrame:
    # выбирает 5000 самых активных рядов для экспериментов
    top_items = (
        train.groupby(["store_nbr", "item_nbr"])["unit_sales"]
        .sum()
        .reset_index()
        .sort_values("unit_sales", ascending=False)
        .head(top_n)
    )

    train_filtered = train.merge(
        top_items[["store_nbr", "item_nbr"]],
        on=["store_nbr", "item_nbr"],
        how="inner"
    )

    return train_filtered
