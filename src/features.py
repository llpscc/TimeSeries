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

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

def add_exogenous(df, items, stores, transactions, oil):
    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(oil, on="date", how="left")
    return df

def add_lags(df: pd.DataFrame, lags=[1, 7, 14, 28]):

    df = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()

    grp = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]

    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)

    df["rolling_mean_7"] = grp.shift(1).rolling(7).mean()
    df["rolling_mean_14"] = grp.shift(1).rolling(14).mean()

    return df

def train_val_split(df, horizon):

    cutoff = df["date"].max() - pd.Timedelta(days=horizon)

    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()

    return train_df, val_df

def prepare_ml_data(df, items, stores, transactions, oil, horizon):

    df = add_exogenous(df, items, stores, transactions, oil)
    df = add_date_features(df)

    train_df, val_df = train_val_split(df, horizon)

    train_df = add_lags(train_df)

    # удаляем строки без лагов
    train_df = train_df.dropna(subset=["lag_28"])

    return train_df, val_df

def prepare_test(df, items, stores, transactions, oil):

    df = add_exogenous(df, items, stores, transactions, oil)
    df = add_date_features(df)

    return df