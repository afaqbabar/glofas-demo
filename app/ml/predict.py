from __future__ import annotations
import os
import joblib
import pandas as pd
from app.ml.features import add_time_features, add_lags

MODELS_DIR = os.getenv("MODELS_DIR", "app/models")

def load_model(name="xgb"):
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)

def prepare_row_for_next_day(history: pd.DataFrame) -> pd.DataFrame:
    """
    history: DataFrame with ['date','dis24_mean'] daily up to 'today'
    Returns a single-row feature frame for predicting 'tomorrow'.
    """
    df = history.copy()
    df = df.sort_values("date")
    # create placeholder date for tomorrow (feature enc uses this)
    tomorrow = pd.to_datetime(df["date"].max()) + pd.Timedelta(days=1)
    df_next = pd.DataFrame({"date": [tomorrow], "dis24_mean": [df["dis24_mean"].iloc[-1]]})
    # Append so lag/rolling can reference last values
    df_all = pd.concat([df, df_next], ignore_index=True)
    df_all = add_time_features(df_all, "date")
    df_all = add_lags(df_all, "dis24_mean", lags=(1,2,3,7,14))
    # return only the last row as features (drop target/date)
    row = df_all.tail(1).drop(columns=["dis24_mean", "date"])
    return row

def predict_next_day(model_name: str, history: pd.DataFrame) -> float:
    model = load_model(model_name)
    X = prepare_row_for_next_day(history)
    return float(model.predict(X)[0])
