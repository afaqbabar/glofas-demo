# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import os
from .glofas_demo import download_bangladesh_if_missing, make_overview_maps, CACHE, BD_GRIB

app = FastAPI(title="GloFAS Bangladesh Demo API", version="0.1.0")

class OverviewReq(BaseModel):
    target_year: int = 2022
    # show None in the docs instead of "string", and treat "string" as empty
    input_path: str | None = Field(default=None, examples=[None])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/download")
def download():
    try:
        path = download_bangladesh_if_missing()
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
        if not os.path.exists(BD_GRIB):
            try:
                download_bangladesh_if_missing()
            except Exception as e:
                raise HTTPException(500, f"Auto-download failed: {e}")
        path = BD_GRIB

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
