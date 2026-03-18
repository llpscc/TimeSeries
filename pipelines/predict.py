import pandas as pd

from src.ml.model_ml import train_lgbm
from src.ml.inference_ml import build_history, recursive_forecast


def run_predict(train_df, val_df, test, X_full, cat_cols):

    full_df = pd.concat([train_df, val_df])

    model = train_lgbm(X_full, full_df["unit_sales"])

    history = build_history(full_df)

    pred_test = recursive_forecast(test, X_full, model, history, cat_cols)

    return pred_test