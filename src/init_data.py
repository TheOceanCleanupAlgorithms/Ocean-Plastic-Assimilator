import netCDF4 as nc
import numpy as np
from shutil import copyfile

from src.io.file_utils import create_folder
from src.assimilation.density_computations import compute_densities
from src.assimilation.density_computations_ensemble import (
    compute_ensemble_densities_over_time,
)
from src.types import AssimilatorConfig, AssimilatorDataPaths, ObservationsType


def recompute_ref_densities(
    config: AssimilatorConfig, ds_in_path: str, ds_out_path: str
):
    ds_out = nc.Dataset(ds_out_path, "r+")

    ds_out["density"][:, :, :] = 0

    T = list(range(ds_out["density"].shape[2]))

    compute_densities(ds_in_path, ds_out_path, T, config.grid_coords, config.cells_area)


def recompute_ensemble_densities(config: AssimilatorConfig, ds_in_path, ds_out_path):
    densities = np.zeros(
        (
            config.size_ensemble,
            config.grid_coords.max_lon_id,
            config.grid_coords.max_lat_id,
            config.max_time,
        )
    )

    ds_parts = nc.Dataset(ds_in_path)
    weights = ds_parts["weight"][:, :]
    parts_lon = ds_parts["lon"][:, :]
    parts_lat = ds_parts["lat"][:, :]
    ds_parts.close()

    compute_ensemble_densities_over_time(
        parts_lon,
        parts_lat,
        densities,
        weights,
        config.cells_area,
        list(range(config.max_time)),
        config.grid_coords,
    )

    ds_out = nc.Dataset(ds_out_path, "r+")
    ds_out["density"][:, :, :, :] = densities
    ds_out.close()


def create_datasets(datapaths: AssimilatorDataPaths, config: AssimilatorConfig):
    ds_densities_ensemble = nc.Dataset(datapaths.ds_densities_ensemble, "w")
    ds_densities_ensemble.createDimension("ensemble", size=config.size_ensemble)
    ds_densities_ensemble.createDimension("lon", size=config.grid_coords.max_lon_id)
    ds_densities_ensemble.createDimension("lat", size=config.grid_coords.max_lat_id)
    ds_densities_ensemble.createDimension("time", size=config.max_time)
    ds_densities_ensemble.createVariable(
        "density", float, dimensions=("ensemble", "lon", "lat", "time")
    )
    ds_densities_ensemble.close()

    if config.observations.type == ObservationsType.from_simulation:
        ds_densities_ref = nc.Dataset(datapaths.ds_densities_ref, "w")
        ds_densities_ref.createDimension("lon", size=config.grid_coords.max_lon_id)
        ds_densities_ref.createDimension("lat", size=config.grid_coords.max_lat_id)
        ds_densities_ref.createDimension("time", size=config.max_time)
        ds_densities_ref.createVariable(
            "density", float, dimensions=("lon", "lat", "time")
        )
        ds_densities_ref.close()

    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "w")
    ds_parts_ensembles.createDimension("p_id", size=config.num_particles_total)
    ds_parts_ensembles.createDimension("time", size=config.max_time)
    ds_parts_ensembles.createDimension("ensemble", size=config.size_ensemble)

    ds_parts_ensembles.createVariable("p_id", int, dimensions=("p_id"))
    ds_parts_ensembles.createVariable("weight", float, dimensions=("ensemble", "p_id"))
    ds_parts_ensembles.createVariable("time", float, dimensions=("time"))
    ds_parts_ensembles.createVariable("lon", float, dimensions=("p_id", "time"))
    ds_parts_ensembles.createVariable("lat", float, dimensions=("p_id", "time"))

    ds_parts_ensembles.close()


def compute_parts_ensemble(datapaths: AssimilatorDataPaths, config: AssimilatorConfig):
    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "a")
    ds_parts_useful = nc.Dataset(datapaths.ds_parts_original)

    # INITIALIZATION : Change total weight
    try:
        weights = ds_parts_useful["weight"][:]
    except IndexError:
        weights = np.array([1] * ds_parts_useful["p_id"].shape[0])
    repeated_weights = np.repeat(weights[np.newaxis, :], config.size_ensemble, axis=0)

    # Normally randomize weights around the new mean defined with NEW_WEIGHT_RATIO
    ds_parts_ensembles["weight"][:, :] = (
        np.array(
            [
                config.initial_mass_multiplicator
                + np.random.randn(config.size_ensemble) * config.ensemble_spread
            ]
        ).transpose()
        * repeated_weights
    )
    # Copy the rest of the vars from the useful set
    ds_parts_ensembles["p_id"][:] = ds_parts_useful["p_id"][:]
    ds_parts_ensembles["time"][:] = ds_parts_useful["time"][
        list(range(config.max_time))
    ]
    ds_parts_ensembles["lon"][:, :] = ds_parts_useful["lon"][
        :, list(range(config.max_time))
    ]
    ds_parts_ensembles["lat"][:, :] = ds_parts_useful["lat"][
        :, list(range(config.max_time))
    ]

    ds_parts_ensembles.close()


def init_data(datapaths: AssimilatorDataPaths, config: AssimilatorConfig):
    create_folder(datapaths.data_dir)
    create_datasets(datapaths, config)

    if config.observations.type == ObservationsType.from_simulation:
        if config.verbose:
            print("Recompute ref densities")
        recompute_ref_densities(
            config, config.observations.ds_reference_path, datapaths.ds_densities_ref
        )
    if config.verbose:
        print("Compute initial particles ensemble")
    compute_parts_ensemble(datapaths, config)

    if config.verbose:
        print("Recompute ensemble densities")
    recompute_ensemble_densities(
        config, datapaths.ds_parts_ensemble, datapaths.ds_densities_ensemble
    )

    copyfile(datapaths.ds_densities_ensemble, datapaths.ds_densities_ensemble.strip(".nc") + "_init.nc")
