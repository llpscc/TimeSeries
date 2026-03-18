import pandas as pd
import numpy as np

from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta

from src.config import HORIZON, SEASON_LENGTH, FREQ
from src.metrics import nwrmsle


def prepare_sf_data(train_5000: pd.DataFrame):

    sf_df = train_5000.copy()

    sf_df["unique_id"] = (
        sf_df["store_nbr"].astype(str) + "_" +
        sf_df["item_nbr"].astype(str)
    )

    sf_df = sf_df.rename(columns={
        "date": "ds",
        "unit_sales": "y"
    })

    sf_df = sf_df[["unique_id", "ds", "y"]].sort_values(["unique_id", "ds"])

    return sf_df


def train_val_split(sf_df):

    max_date = sf_df["ds"].max()
    cutoff = max_date - pd.Timedelta(days=HORIZON)

    train_sf = sf_df[sf_df["ds"] <= cutoff]
    val_sf = sf_df[sf_df["ds"] > cutoff]

    return train_sf, val_sf


def run_baselines(train_sf, val_sf, items):

    models = [
        Naive(),
        SeasonalNaive(season_length=SEASON_LENGTH),
        AutoETS(season_length=SEASON_LENGTH),
        AutoTheta(season_length=SEASON_LENGTH)
    ]

    sf = StatsForecast(
        models=models,
        freq=FREQ,
        n_jobs=-1,
        verbose=True
    )

    forecast = sf.forecast(df=train_sf, h=HORIZON)

def merge_sf_forecast(forecast, val_sf, items):
    eval_df = forecast.merge(val_sf, on=["unique_id", "ds"], how="left")
    eval_df["y"] = eval_df["y"].fillna(0)

    eval_df["item_nbr"] = eval_df["unique_id"].str.split("_").str[1].astype(int)

    eval_df = eval_df.merge(
        items[["item_nbr", "perishable"]],
        on="item_nbr",
        how="left"
    )

    eval_df["y"] = np.clip(eval_df["y"], 0, None)

    for model in ["Naive", "SeasonalNaive", "AutoETS", "AutoTheta"]:
        eval_df[model] = np.clip(eval_df[model], 0, None)

    return eval_df        

def compute_sf_metrics(eval_df):
    eval_df["weight"] = np.where(eval_df["perishable"] == 1, 1.25, 1.0)

    metrics = {}

    for model in ["Naive", "SeasonalNaive", "AutoETS", "AutoTheta"]:
        score = nwrmsle(
            eval_df["y"].values,
            eval_df[model].values,
            eval_df["weight"].values
        )
        metrics[model] = score

    results = pd.DataFrame(
        metrics.items(),
        columns=["Model", "NWRMSLE"]
    ).sort_values("NWRMSLE")

    return results
