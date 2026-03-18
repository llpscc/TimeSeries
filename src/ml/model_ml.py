import lightgbm as lgb


def prepare_data(train_df, val_df):

    X_train = train_df.drop(columns=["unit_sales", "date", "unique_id", 'id'], errors="ignore")
    y_train = train_df["unit_sales"]

    X_val = val_df.drop(columns=["unit_sales", "unique_id", 'id'], errors="ignore")
    y_val = val_df["unit_sales"]

    return X_train, y_train, X_val, y_val


def set_categories(X_train, X_val, cat_cols):

    for col in cat_cols:
        X_train[col] = X_train[col].astype("category")
        X_val[col] = X_val[col].astype("category")

    return X_train, X_val


def train_lgbm(X_train, y_train, cat_cols):

    model = lgb.LGBMRegressor(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=128,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbosity=-1,
        force_col_wise=True
    )

    model.fit(X_train, y_train, categorical_feature=cat_cols)

    return model
