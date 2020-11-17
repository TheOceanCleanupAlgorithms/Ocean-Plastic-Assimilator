import os
import netCDF4 as nc
import numpy as np
from src.assimilation.heavy_computations import computeDensitiesParallel
from src.assimilation.heavy_computations_ensembles import (
    computeEnsembleDensitiesOverTime,
)
from sim_vars import (
    LONGITUDES,
    LATITUDES,
    TIMES,
    data_dir_path,
    ds_densities_ref_path,
    ds_parts_useful_path,
    ds_parts_ref_path,
    ds_parts_ensembles_path,
    NB_PARTS,
    NB_ENSEMBLES,
    ds_densities_ensembles_path,
    INIT_STD_DEV,
    NEW_WEIGHT_RATIO,
)


def recomputeRefDensities(ds_in_path, ds_out_path):
    ds_out = nc.Dataset(ds_out_path, "r+")

    ds_out["density"][:, :, :] = np.zeros((LONGITUDES, LATITUDES, TIMES))

    T = list(range(TIMES))

    computeDensitiesParallel(ds_in_path, ds_out_path, T, nbThreads=1)


def recomputeEnsembleDensities(ds_in_path, ds_out_path):
    densities = np.zeros((NB_ENSEMBLES, LONGITUDES, LATITUDES, TIMES))

    ds_parts = nc.Dataset(ds_in_path)
    weights = ds_parts["weight"][:, :]
    parts_lon = ds_parts["lon"][:, :]
    parts_lat = ds_parts["lat"][:, :]
    ds_parts.close()

    computeEnsembleDensitiesOverTime(
        parts_lon, parts_lat, densities, weights, list(range(TIMES))
    )

    ds_out = nc.Dataset(ds_out_path, "r+")
    ds_out["density"][:, :, :, :] = densities
    ds_out.close()


def init_data():
    # Create dataset files
    try:
        os.mkdir(data_dir_path)
    except OSError:
        print(
            "Directory",
            data_dir_path,
            "couldn't be created, perhaps it already exists?",
        )
        exit()
    else:
        print("Successfully created", data_dir_path)

        ds_densities_ensembles = nc.Dataset(ds_densities_ensembles_path, "w")
        ds_densities_ensembles.createDimension("ensemble", size=NB_ENSEMBLES)
        ds_densities_ensembles.createDimension("lon", size=LONGITUDES)
        ds_densities_ensembles.createDimension("lat", size=LATITUDES)
        ds_densities_ensembles.createDimension("time", size=TIMES)
        ds_densities_ensembles.createVariable(
            "density", float, dimensions=("ensemble", "lon", "lat", "time")
        )
        ds_densities_ensembles.close()

        ds_densities_ref = nc.Dataset(ds_densities_ref_path, "w")
        ds_densities_ref.createDimension("lon", size=LONGITUDES)
        ds_densities_ref.createDimension("lat", size=LATITUDES)
        ds_densities_ref.createDimension("time", size=TIMES)
        ds_densities_ref.createVariable(
            "density", float, dimensions=("lon", "lat", "time")
        )
        ds_densities_ref.close()

        ds_parts_ensembles = nc.Dataset(ds_parts_ensembles_path, "w")
        ds_parts_ensembles.createDimension("x", size=NB_PARTS)
        ds_parts_ensembles.createDimension("time", size=TIMES)
        ds_parts_ensembles.createDimension("ensemble", size=NB_ENSEMBLES)

        ds_parts_ensembles.createVariable("id", int, dimensions=("x"))
        ds_parts_ensembles.createVariable("weight", float, dimensions=("ensemble", "x"))
        ds_parts_ensembles.createVariable("time", float, dimensions=("time"))
        ds_parts_ensembles.createVariable("lon", float, dimensions=("x", "time"))
        ds_parts_ensembles.createVariable("lat", float, dimensions=("x", "time"))

        ds_parts_ensembles.close()

    print("Recompute ref densities")
    os.system('say -v "Alex" "Beginning computation of densities"')
    recomputeRefDensities(ds_parts_ref_path, ds_densities_ref_path)
    os.system('say -v "Alex" "Computation of reference densities is over."')

    ds_parts_ensembles = nc.Dataset(ds_parts_ensembles_path, "a")
    ds_parts_useful = nc.Dataset(ds_parts_useful_path)

    # INITIALIZATION : Change total weight
    weights = ds_parts_useful["weight"][:]
    repeated_weights = np.repeat(weights[np.newaxis, :], NB_ENSEMBLES, axis=0)
    # Normally randomize weights around the new mean defined with NEW_WEIGHT_RATIO
    ds_parts_ensembles["weight"][:, :] = (
        np.array(
            [np.random.randn(NB_ENSEMBLES) * INIT_STD_DEV + NEW_WEIGHT_RATIO]
        ).transpose()
        * repeated_weights
    )
    # Copy the rest of the vars from the useful set
    ds_parts_ensembles["id"][:] = ds_parts_useful["id"][:]
    ds_parts_ensembles["time"][:] = ds_parts_useful["time"][list(range(TIMES))]
    ds_parts_ensembles["lon"][:, :] = ds_parts_useful["lon"][:, list(range(TIMES))]
    ds_parts_ensembles["lat"][:, :] = ds_parts_useful["lat"][:, list(range(TIMES))]

    ds_parts_ensembles.close()

    print("Recompute ensemble densities")
    recomputeEnsembleDensities(ds_parts_ensembles_path, ds_densities_ensembles_path)

    os.system('say -v "Alex" "Computation of ensemble densities is over."')


if __name__ == "__main__":
    init_data()
