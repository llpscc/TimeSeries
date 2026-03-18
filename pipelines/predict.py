import pandas as pd

from src.models import train_lgbm
from src.inference import build_history, recursive_pred


def run_predict(train_df, val_df, test, X_full, cat_cols):

    full_df = pd.concat([train_df, val_df])

    model = train_lgbm(X_full, full_df["unit_sales"])

    history = build_history(full_df)

    pred_test = recursive_pred(test, X_full, model, history, cat_cols)

    return pred_test