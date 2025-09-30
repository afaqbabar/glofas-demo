# climetlab_glofas_demo/__init__.py
import climetlab as cml
from climetlab import Dataset

# Helper: build CDS request dict
def _hist_request(years, months, days, area, system_version="version_4_0", model="lisflood"):
    return {
        "system_version": system_version,
        "hydrological_model": model,
        "product_type": "consolidated",
        "variable": "river_discharge_in_the_last_24_hours",
        "hyear": [str(y) for y in years],
        "hmonth": [f"{m:02d}" for m in months],
        "hday": [f"{d:02d}" for d in days],
        "data_format": "grib2",
        "download_format": "unarchived",
        "area": area,     # [N, W, S, E]
    }

class PKAugustHistorical(Dataset):
    """Pakistan August (2012–2024) GloFAS historical — matches your demo defaults."""
    name = "glofas-bd-june-historical"
    home_page = "https://github.com/afaqbabar/glofas-demo"
    description = "Convenience dataset for Pakistan August GloFAS historical tiles."

    def __init__(self, start=2012, end=2024, area=(37.5, 60.0, 23.0, 77.5)):
        years = range(int(start), int(end) + 1)
        req = _hist_request(years=years, months=[6], days=range(1, 31), area=list(area))
        self.source = cml.load_source("cds", "cems-glofas-historical", req)

    def to_xarray(self, **kwargs):
        return self.source.to_xarray(**kwargs)

class GlofasHistorical(Dataset):
    """Generic GloFAS historical loader with friendly params."""
    name = "glofas-generic-historical"
    home_page = "https://github.com/afaqbabar/glofas-demo"
    description = "Generic GloFAS historical via CDS/EWDS (consolidated)."

    def __init__(self,
                 start=2012, end=2024,
                 months=(8,), days=None,
                 area=(30, 85, 20, 95),
                 system_version="version_4_0",
                 hydrological_model="lisflood"):
        years = range(int(start), int(end) + 1)
        if days is None:
            days = range(1, 32)  # safe; CDS ignores invalid dates per month
        req = _hist_request(years=years,
                            months=[int(m) for m in months],
                            days=days,
                            area=list(area),
                            system_version=system_version,
                            model=hydrological_model)
        self.source = cml.load_source("cds", "cems-glofas-historical", req)

    def to_xarray(self, **kwargs):
        return self.source.to_xarray(**kwargs)
