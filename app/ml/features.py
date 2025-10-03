from __future__ import annotations
import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    doy = dt.dt.dayofyear
    df["month"] = dt.dt.month
    df["doy_sin"] = np.sin(2 * np.pi * doy / 366)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 366)
    return df

def add_lags(df: pd.DataFrame, y_col: str = "dis24_mean", lags=(1,2,3,7,14)) -> pd.DataFrame:
    df = df.copy()
    for L in lags:
        df[f"{y_col}_lag{L}"] = df[y_col].shift(L)
    df[f"{y_col}_roll7"] = df[y_col].rolling(7).mean()
    df[f"{y_col}_roll14"] = df[y_col].rolling(14).mean()
    return df

def train_test_split_time(df: pd.DataFrame, date_col="date", split_date="2022-01-01"):
    df = df.sort_values(date_col)
    train = df[df[date_col] < split_date].dropna()
    test  = df[df[date_col] >= split_date].dropna()
    return train, test

def xy(df: pd.DataFrame, y_col="dis24_mean"):
    X = df.drop(columns=[y_col, "date"])
    y = df[y_col].values
    return X, y
