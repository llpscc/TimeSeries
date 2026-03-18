from src.ml.model_ml import train_lgbm
from src.ml.inference_ml import build_history, recursive_pred
from src.metrics import nwrmsle


def run_train(train_df, val_df, X_train, y_val, cat_cols):

    model = train_lgbm(X_train, train_df["unit_sales"])

    history = build_history(train_df)

    pred_val = recursive_pred(val_df, X_train, model, history, cat_cols)

    score = nwrmsle(
        y_val.values,
        pred_val["pred"].values,
        pred_val["perishable"].values
    )

    return score
