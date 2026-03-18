# подготовка данных для TFT

import numpy as np
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer


def prepare_dl_dataset(dl_df, HORIZON, categorical_cols):
    # копия датасета
    dl_df = dl_df.copy()

    # создаем признак id ряда 
    dl_df["series_id"] = (
        dl_df["store_nbr"].astype(str) + "_" +
        dl_df["item_nbr"].astype(str)
    )

    # сортируем данные по хронологии
    dl_df = dl_df.sort_values(["series_id", "date"]).reset_index(drop=True)

    # присваиваем датам индексы, обрезаем отрицательные значения
    dl_df["time_idx"] = (dl_df["date"] - dl_df["date"].min()).dt.days
    dl_df["unit_sales"] = dl_df["unit_sales"].clip(lower=0)

    max_time_idx = dl_df["time_idx"].max()
    training_cutoff = max_time_idx - HORIZON

    categorical_cols = [
        "store_nbr","item_nbr","family","class",
        "city","cluster","day_of_week","week_of_year",
        "month","onpromotion","perishable"
    ]

    for col in categorical_cols:
        dl_df[col] = dl_df[col].astype(str)

    dl_df = dl_df.rename(columns={"type": "store_type"})

    # очистка
    dl_df = dl_df.replace([np.inf, -np.inf], np.nan)
    dl_df["transactions"] = dl_df["transactions"].fillna(0)
    dl_df["dcoilwtico"] = dl_df["dcoilwtico"].ffill().bfill()

    return dl_df, training_cutoff


def build_tft_dataset(dl_df, training_cutoff, HORIZON, max_encoder_length, static_categoricals, time_varying_known_reals, time_varying_unknown_reals):

    # горизонт прогнозирования
    max_prediction_length = HORIZON
    # создаем датасет
    training = TimeSeriesDataSet(
         # используем train данные
        dl_df[dl_df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="unit_sales",
        group_ids=["series_id"],

        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,

        # группы признаков
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        
        # нормализуем таргет
        target_normalizer=GroupNormalizer(
            groups=["series_id"],
            transformation="softplus"
        ),
        # дополнительные индекс, масштаб ряда и длина истории
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,

        allow_missing_timesteps=True,
    )
    # создаем валидационный датасет с теми же настройками
    validation = TimeSeriesDataSet.from_dataset(
        training,
        dl_df,
        predict=True,
        stop_randomization=True
    )

    return training, validation