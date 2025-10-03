# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import os
from statistics import mean, pstdev

from .glofas_demo import download_pakistan_if_missing, make_overview_maps, CACHE, PK_GRIB

from app.ml.train_models import main as train_main
from app.ml.predict import predict_next_day

from app.ml.utils import load_thresholds



app = FastAPI(title="GloFAS Pakistan Demo API", version="0.1.0")

class OverviewReq(BaseModel):
    target_year: int = 2025
    # show None in the docs instead of "string", and treat "string" as empty
    input_path: str | None = Field(default=None, examples=[None])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/download")
def download():
    try:
        path = download_pakistan_if_missing()
        size = os.path.getsize(path) if os.path.exists(path) else 0
        return {"downloaded": os.path.exists(path) and size > 0, "path": path, "size_bytes": size}
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")

@app.get("/files")
def list_cache():
    files = sorted([os.path.join(CACHE, f) for f in os.listdir(CACHE)])
    return {"cache": files}

@app.post("/overview")
def overview(req: OverviewReq):
    # normalize input_path (Swagger often sends "string")
    path = req.input_path
    if path in (None, "", "string"):
        # ensure the canonical GRIB exists
        if not os.path.exists(PK_GRIB):
            try:
                download_pakistan_if_missing()
            except Exception as e:
                raise HTTPException(500, f"Auto-download failed: {e}")
        path = PK_GRIB

    if not os.path.exists(path):
        raise HTTPException(404, f"Input GRIB not found: {path}")

    # helpful server-side log
    print(f"[overview] using GRIB: {path}  target_year={req.target_year}")

    try:
        out = make_overview_maps(input_path=path, target_year=req.target_year)
        return out
    except Exception as e:
        raise HTTPException(500, f"Overview generation failed: {e}")

@app.get("/image/{filename}")
def image(filename: str):
    path = os.path.join(CACHE, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/png")



# app = FastAPI()

class TrainReq(BaseModel):
    start: int = 2015
    end: int = 2024
    months: list[int] = [8]
    area: tuple[float,float,float,float] = (37.5,60.0,23.0,77.5)
    split_date: str = "2022-01-01"

@app.post("/ml/train")
def train(req: TrainReq):
    # quick-and-dirty: call the CLI entry by constructing argv
    import sys
    argv_save = sys.argv
    sys.argv = [
        "train_models.py", "--start", str(req.start), "--end", str(req.end),
        "--months", *[str(m) for m in req.months],
        "--area", *[str(x) for x in req.area],
        "--split-date", req.split_date
    ]
    try:
        train_main()
        return {"status": "ok"}
    finally:
        sys.argv = argv_save

class PredictReq(BaseModel):
    model_name: str = "xgb"
    # Expect a minimal recent history to compute lags/rolling
    # [{ "date":"2024-08-01", "dis24_mean": 123.4 }, ...]
    history: list[dict]

@app.post("/ml/predict-next-day")
def predict(req: PredictReq):
    df = pd.DataFrame(req.history)
    yhat = predict_next_day(req.model_name, df)

    # context from thresholds
    th = load_thresholds() or {}
    p95 = th.get("p95_train")
    p99 = th.get("p99_train")
    mu  = th.get("mean_train")
    sd  = th.get("std_train")

    z = None if (mu is None or sd in (None, 0)) else (yhat - mu) / sd
    flags = {
        "ge_p95": (p95 is not None and yhat >= p95),
        "ge_p99": (p99 is not None and yhat >= p99),
    }

    return {
        "model": req.model_name,
        "prediction_dis24_mean": float(yhat),
        "zscore_vs_train": None if z is None else float(z),
        "p95_train": p95,
        "p99_train": p99,
        "flags": flags
    }