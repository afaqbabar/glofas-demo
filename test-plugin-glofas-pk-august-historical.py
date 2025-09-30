import climetlab as cml
ds = cml.load_dataset("glofas-pk-august-historical")
xr = ds.to_xarray(engine="cfgrib")
print("Dims:", xr.dims)
print("Vars:", list(xr.data_vars))

da = xr['dis24']
print(float(da.mean().values), float(da.max().values))   # overall mean/max