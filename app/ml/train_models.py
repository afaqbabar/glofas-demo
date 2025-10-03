from __future__ import annotations
import argparse
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import joblib

# Light imports first for Pi; heavier ones later
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import climetlab as cml  # you documented this usage in README
from app.ml.features import add_time_features, add_lags, train_test_split_time, xy

MODELS_DIR = os.getenv("MODELS_DIR", "app/models")
os.makedirs(MODELS_DIR, exist_ok=True)

def load_glofas_series(
    start=2015, end=2024, months=(8,), area=(37.5, 60.0, 23.0, 77.5), var_name="dis24"
) -> pd.DataFrame:
    """
    Load historical GloFAS via your CliMetLab entrypoint and reduce
    the region to a single daily series (spatial mean of discharge).
    """
    xr_ds: xr.Dataset = cml.load_dataset(
        "glofas-generic-historical",
        start=int(start), end=int(end), months=tuple(months), area=tuple(area)
    ).to_xarray(engine="cfgrib")
    # Variable often called 'dis24' (as noted in your README)
    if var_name not in xr_ds:
        # try common fallbacks
        for cand in ["dis24", "dis"]:
            if cand in xr_ds:
                var_name = cand
                break
    da = xr_ds[var_name]
    # spatial mean over lat/lon dims
    da_mean = da.mean(dim=[d for d in da.dims if d not in ("time",)], skipna=True)
    df = da_mean.to_dataframe(name="dis24_mean").reset_index()  # columns: time, dis24_mean
    df = df.rename(columns={"time": "date"})
    # Ensure daily frequency and sorted
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").dropna()
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df, "date")
    df = add_lags(df, "dis24_mean", lags=(1,2,3,7,14))
    return df

def eval_and_save(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    out_path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(model, out_path)
    return {"name": name, "rmse": rmse, "r2": r2, "path": out_path}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2015)
    p.add_argument("--end", type=int, default=2024)
    p.add_argument("--months", type=int, nargs="+", default=[8])  # August by default (your PK example)
    p.add_argument("--area", type=float, nargs=4, default=[37.5, 60.0, 23.0, 77.5], help="[N W S E]")
    p.add_argument("--split-date", type=str, default="2022-01-01")
    args = p.parse_args()

    df = load_glofas_series(args.start, args.end, tuple(args.months), tuple(args.area))
    df = build_features(df)
    train, test = train_test_split_time(df, split_date=args.split_date)
    Xtr, ytr = xy(train)
    Xte, yte = xy(test)

    results = []
    results.append(eval_and_save("linreg", LinearRegression(), Xtr, ytr, Xte, yte))
    results.append(eval_and_save("rf", RandomForestRegressor(
        n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
    ), Xtr, ytr, Xte, yte))

    if HAS_XGB:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        xgb = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9,
            colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", random_state=42
        )
        results.append(eval_and_save("xgb", xgb, Xtr, ytr, Xte, yte))
    else:
        results.append({"name": "xgb", "rmse": None, "r2": None, "path": None, "note": "xgboost not installed"})

    # Save a quick CSV report
    pd.DataFrame(results).to_csv(os.path.join(MODELS_DIR, "metrics.csv"), index=False)
    for row in results:
        print(row)

    # --- thresholds & summary ---
    p95 = float(train["dis24_mean"].quantile(0.95))
    p99 = float(train["dis24_mean"].quantile(0.99))
    summary = {
        "split_date": args.split_date,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "p95_train": p95,
        "p99_train": p99,
        "mean_train": float(train["dis24_mean"].mean()),
        "std_train": float(train["dis24_mean"].std())
    }
    pd.Series(summary).to_json(os.path.join(MODELS_DIR, "thresholds.json"))

if __name__ == "__main__":
    main()
