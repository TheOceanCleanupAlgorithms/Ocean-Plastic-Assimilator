import netCDF4 as nc
import numpy as np


def exportParticlesToDataset(ds_name, lons, lats, epsilon, A):
    NB_PARTS = lons.shape[0]
    NB_ITER = lons.shape[1]

    ds_parts_path = (
        "dispersion_double_gyre/"
        + "parts_"
        + ds_name
        + "_"
        + str(NB_PARTS)
        + "_"
        + str(NB_ITER)
        + "_eps_"
        + str(epsilon)
        + "_A_"
        + str(A)
        + ".nc"
    )
    ds_parts = nc.Dataset(ds_parts_path, "w")

    ds_parts.createDimension("x", size=NB_PARTS)
    ds_parts.createDimension("time", size=NB_ITER)
    dsp_var_id = ds_parts.createVariable("p_id", int, dimensions=("x"))
    dsp_var_weight = ds_parts.createVariable("weight", float, dimensions=("x"))
    dsp_var_time = ds_parts.createVariable("time", float, dimensions=("time"))
    dsp_var_lon = ds_parts.createVariable("lon", float, dimensions=("x", "time"))
    dsp_var_lat = ds_parts.createVariable("lat", float, dimensions=("x", "time"))

    dsp_var_lon[:, :] = lons
    dsp_var_lat[:, :] = lats
    dsp_var_time[:] = list(range(NB_ITER))
    dsp_var_id[:] = list(range(NB_PARTS))
    dsp_var_weight[:] = np.array(NB_PARTS * [1.0])

    ds_parts.close()
