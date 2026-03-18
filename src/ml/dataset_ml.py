
import pandas as pd

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # добавляет новые признаки на основе даты
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df

def add_exogenous(df, items, stores, transactions, oil):
    
    # добавляет экзогенные признаки
    df = df.merge(items, on="item_nbr", how="left")
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(oil, on="date", how="left")
    return df

def add_lags(df: pd.DataFrame, lags=[1, 7, 14, 28]):

    # добавляет лаги и скользящие средние
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).copy()

    grp = df.groupby(["store_nbr", "item_nbr"])["unit_sales"]

    for lag in lags:
        df[f"lag_{lag}"] = grp.shift(lag)

    df["rolling_mean_7"] = (
        grp.shift(1)
        .groupby([df["store_nbr"], df["item_nbr"]])
        .rolling(7)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    df["rolling_mean_14"] = (
        grp.shift(1)
        .groupby([df["store_nbr"], df["item_nbr"]])
        .rolling(14)
        .mean()
        .reset_index(level=[0,1], drop=True)
    )

    return df

def train_val_split(df, horizon):
    # делит данные на train и val
    cutoff = df["date"].max() - pd.Timedelta(days=horizon)

    train_df = df[df["date"] <= cutoff].copy()
    val_df = df[df["date"] > cutoff].copy()

    return train_df, val_df

def prepare_ml_data(df, items, stores, transactions, oil, horizon):
    # соединяет функции add_exogenous, add_date_features, train_val_split и add_lags
    
    df = add_exogenous(df, items, stores, transactions, oil)
    df = add_date_features(df)

    train_df, val_df = train_val_split(df, horizon)

    train_df = add_lags(train_df)

    # удаляем строки без лагов
    train_df = train_df.dropna(subset=["lag_28"])

    return train_df, val_df

def prepare_test(df, items, stores, transactions, oil):
    # отдельная функция для добавления признаков в тест без лагов и роллингов
    df['onpromotion'] = df['onpromotion'].astype(int)

    df = add_exogenous(df, items, stores, transactions, oil)
    df = add_date_features(df)

    return df

def test_categories(df, cat_cols):
    # преобразование столбцов в типа категория
    for col in cat_cols:
        df[col] = df[col].astype("category")

    return df
