import os, cdsapi, xarray as xr, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cache"))
os.makedirs(CACHE, exist_ok=True)

BD_GRIB = os.path.join(CACHE, "glofas-2012_2022_BD.grib")

def download_bangladesh_if_missing() -> str:
    """Download June 2012–2022 Bangladesh GRIB into cache/, idempotent."""
    if os.path.isfile(BD_GRIB) and os.path.getsize(BD_GRIB) > 0:
        return BD_GRIB
    c = cdsapi.Client()
    c.retrieve(
        "cems-glofas-historical",
        {
            "system_version": ["version_4_0"],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": [f"{y}" for y in range(2012, 2023)],
            "hmonth": ["06"],
            "hday": [f"{d:02d}" for d in range(1, 31)],
            "data_format": "grib2",
            "download_format": "unarchived",
            "area": [30, 85, 20, 95]  # N,W,S,E (Bangladesh bbox)
        }
    ).download(BD_GRIB)
    return BD_GRIB

def _pick_var(ds):
    return "dis24" if "dis24" in ds.data_vars else ("mdis24" if "mdis24" in ds.data_vars else list(ds.data_vars)[0])

def make_overview_maps(input_path: str = BD_GRIB, target_year: int = 2022):
    """Create two PNGs in cache/ and return their absolute paths."""
    ds = xr.open_dataset(input_path, engine="cfgrib")
    var = _pick_var(ds)
    lat = ds.latitude.values; lon = ds.longitude.values
    LON, LAT = np.meshgrid(lon, lat)

    # 1) Mean across all times in the file
    mean_all = ds[var].mean(dim="time")
    out1 = os.path.join(CACHE, "mean_june_all_years.png")
    plt.figure(figsize=(7.8, 6))
    im = plt.pcolormesh(LON, LAT, mean_all.values, shading="auto")
    cb = plt.colorbar(im, shrink=0.85); cb.set_label("Mean discharge (m³/s, June)")
    plt.title("GloFAS (historical) — Bangladesh\nMean June discharge (2012–2022)")
    plt.xlabel("Lon"); plt.ylabel("Lat"); plt.tight_layout()
    plt.savefig(out1, dpi=150); plt.close()

    # 2) Triptych: q95 (2012–2021), June target_year mean, exceedance fraction
    is_tgt = ds["time"].dt.year == target_year
    if is_tgt.sum().item() == 0:
        ds.close()
        raise ValueError(f"No data for target_year={target_year} in {input_path}")
    ref = ds.where(~is_tgt, drop=True)
    tgt = ds.where( is_tgt, drop=True)
    clim_q95 = ref[var].quantile(0.95, dim="time")
    jun_mean = tgt[var].mean(dim="time")
    exceed = (tgt[var] > clim_q95).mean(dim="time")

    out2 = os.path.join(CACHE, f"overview_triptych_{target_year}.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    im0 = axes[0].pcolormesh(LON, LAT, clim_q95.values, shading="auto")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label("m³/s")
    axes[0].set_title("Climatology q95"); axes[0].set_xlabel("Lon"); axes[0].set_ylabel("Lat")

    im1 = axes[1].pcolormesh(LON, LAT, jun_mean.values, shading="auto")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label("m³/s")
    axes[1].set_title(f"June {target_year} mean"); axes[1].set_xlabel("Lon"); axes[1].set_ylabel("Lat")

    im2 = axes[2].pcolormesh(LON, LAT, exceed.values, shading="auto", vmin=0, vmax=1)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label("Fraction of days")
    axes[2].set_title(f"June {target_year} exceedance of q95"); axes[2].set_xlabel("Lon"); axes[2].set_ylabel("Lat")

    plt.suptitle("GloFAS Bangladesh — overview", y=1.02)
    plt.savefig(out2, dpi=150); plt.close()
    ds.close()
    return {"mean_png": out1, "triptych_png": out2}