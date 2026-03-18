import numpy as np
import pandas as pd
from tqdm import tqdm


def build_history(train_df):
    # строит историю по тренировочным данным для построения лагов и роллингов для теста 
    return (
        train_df.sort_values("date")
        .groupby(["store_nbr", "item_nbr"])["unit_sales"]
        .apply(lambda x: list(x.tail(28)))
        .to_dict()
    )


def recursive_forecast(X_test, X_train, model, history, cat_cols):
    
    history = history.copy()
    predictions = []
    
    # идем по всем датам в test
    for d in tqdm(sorted(X_test["date"].unique()), desc="Predicting"):

        day_df = X_test[X_test["date"] == d].copy()

        lag1, lag7, lag14, lag28 = [], [], [], []
        roll7, roll14 = [], []
        # считаем и добавляем лаги и роллинги
        for row in day_df.itertuples():

            key = (row.store_nbr, row.item_nbr)
            hist = history.get(key, [])

            if len(hist) < 28:
                hist = [0] * (28 - len(hist)) + hist

            lag1.append(hist[-1])
            lag7.append(hist[-7])
            lag14.append(hist[-14])
            lag28.append(hist[-28])

            roll7.append(np.mean(hist[-7:]))
            roll14.append(np.mean(hist[-14:]))

        day_df["lag_1"] = lag1
        day_df["lag_7"] = lag7
        day_df["lag_14"] = lag14
        day_df["lag_28"] = lag28

        day_df["rolling_mean_7"] = roll7
        day_df["rolling_mean_14"] = roll14

        X_day = day_df[X_train.columns].copy()
        # меняем тип данных на категорию
        for col in cat_cols:
            X_day[col] = X_day[col].astype("category")
            X_day[col] = X_day[col].cat.set_categories(
                X_train[col].cat.categories
            )
        # предсказание отдельно для каждого дня
        pred = model.predict(X_day)
        pred = np.clip(pred, 0, None)

        day_df["pred"] = pred
        
        # обновление истории
        for row in day_df.itertuples():
            key = (row.store_nbr, row.item_nbr)

            if key not in history:
                history[key] = []

            history[key].append(row.pred)
            history[key] = history[key][-28:]

        predictions.append(day_df)

    return pd.concat(predictions, ignore_index=True)
