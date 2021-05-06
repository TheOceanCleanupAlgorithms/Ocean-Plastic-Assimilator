from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import netCDF4 as nc
from datetime import datetime

import pandas as pd

from src.assimilation.start_simulation import start_simulation
from src.init_data import init_data
from src.types import (
    AssimilatorConfig,
    AssimilatorDataPaths,
    ObservationsFromCSVConfig,
    ObservationsFromSimulationConfig,
    ObservationsType,
    RectGridCoords,
)

PARTICLES_DS_VARIABLES = ["p_id", "lon", "lat", "time"]


def run_assimilator(
    particles_dataset_path: str,
    observations_type: str,
    observations_source_path: str,
    assimilation_domain_coords: Tuple[float, float, float, float],
    assimilation_grid_size: Tuple[int, int],
    size_ensemble: int,
    initial_ensemble_spread: float,
    t_start: int,
    t_end: int,
    initial_mass_multiplicator: float = 1,
    radius_observation: int = np.infty,
    reinit_spreading: float = 0,
    metrics_plot_period: int = 10,
    observations_error_percent: Optional[float] = None,
    observation_locations: Optional[List[Tuple[int, int]]] = None,
    measure_resolution: Optional[float] = None,
    simulation_name: Optional[str] = None,
    csv_reader_options: Dict = {},
    verbose: bool = False,
    computations_data_dir: Optional[str] = None,
    cells_area: Optional[np.ndarray] = None,
):
    """
    Start the ocean plastic assimilator

    :param particles_dataset_path: path to the netcdf file containing particles positions in time.
        Usually an output from a dispersion model
        See docs/data_requirements.md for format requirements
    :param observations_type: Either 'from_simulation' or 'from_csv'
        from_simulation is for when you want to sample observations from another particles netcdf file
        from_csv is for when you want to retrieve observations recorder in a csv file
        See docs/data_requirements.md for format requirements
    :param observations_source_path: Path to the csv or netcdf file to retrieve or sample observations.
    :param assimilation_domain_coords: Coordinates x1,y1,x2,y2 of the rectangle delimitating the area of assimilation
    :param assimilation_grid_size: Tuple n,p with n the number of grid cells on the horizontal axis, y on the vertical axis.
    :param size_ensemble: Size of the ensemble
    :param initial_ensemble_spread: Initial standard deviation / spread of the ensemble
    :param t_start: Time index of the input particles file at which to start assimilating
    :param t_end: Time index at which to stop assimilating
    :param initial_mass_multiplicator: Multiplicator of the initial total mass of the simulation observatations are assimilated in.
        Defaults to 1.
    :param radius_observation: Radius of local assimilation around each observation point. Experimental.
        Defaults to infinity, which is equivalent to using no localization.
    :param metrics_plot_period: Period at which a png file showing several graphs will be output.
        Defaults to 10. Because why not.
    :param observations_error_percent: When using observations sampled from another simulation, allows to simulate an observation error.
        Defaults to None, should always be defined and non-zero in case observation type is from_simulation.
    :param observation_locations: Coordinates in the assimilation grid of the observation points used to sample observations.
        Defaults to None, should always be defined in case observation type is from_simulation.
    :param measure_resolution: Simulated measure resolution when sampling observations.
        Defaults to None, should always be defined and non-zero in case observation type is from_simulation.
    :param simulation_name: Name to give to your simulation and the folder containing its files.
        Defaults to the time at which the simulation is started.
    :param csv_reader_options: Options transmitted to pandas.read_csv when using obs_type = from_csv
        Defaults to empty dict.
    :param verbose: Set to False if you want to see less messages.
        Defaults to True.
    """

    try:
        obs_type_enum = ObservationsType[observations_type]
    except KeyError:
        raise ValueError(
            f"Invalid argument observations_type; must be one of "
            f"{set(obs_type.name for obs_type in ObservationsType)}."
        )

    if simulation_name is None:
        simulation_name = datetime.now().strftime("%y%m%d%H%M%S")
    
    if cells_area is None:
        cells_area = np.ones(assimilation_grid_size)

    # If paths are not defined, create directories here
    all_outputs_dir = "outputs/"
    computations_data_dir = "data/" if computations_data_dir is None else computations_data_dir
    Path(all_outputs_dir).mkdir(exist_ok=True)
    Path(computations_data_dir).mkdir(exist_ok=True)

    data_dir = f"{computations_data_dir}data_{simulation_name}/"
    metrics_dir = f"{all_outputs_dir}output_{simulation_name}/"

    ds_particles = nc.Dataset(particles_dataset_path)

    for var in PARTICLES_DS_VARIABLES:
        assert (
            var in ds_particles.variables
        ), f"missing variable '{var}' in {particles_dataset_path}"

    if obs_type_enum == ObservationsType.from_simulation:
        ds_particles_ref = nc.Dataset(observations_source_path)
        for var in PARTICLES_DS_VARIABLES:
            assert (
                var in ds_particles_ref.variables
            ), f"missing variable '{var}' in {observations_source_path}"

    datapaths = AssimilatorDataPaths(
        metrics_dir=metrics_dir,
        data_dir=data_dir,
        ds_parts_original=particles_dataset_path,
        ds_parts_ensemble=f"{data_dir}parts_ensemble_{simulation_name}.nc",
        ds_densities_ensemble=f"{data_dir}densities_ensemble_{simulation_name}.nc",
        ds_densities_ref=f"{data_dir}densities_ref_{simulation_name}.nc",
    )

    num_parts = ds_particles["p_id"].size

    if measure_resolution is None:
        measure_resolution = (
            num_parts / (assimilation_grid_size[0] * assimilation_grid_size[1]) * 0.01
        )

    try:
        obs_type_enum = ObservationsType[observations_type]
    except KeyError:
        raise ValueError(
            f"Invalid argument observations_type; must be one of "
            f"{set(obs_type.name for obs_type in ObservationsType)}."
        )

    if obs_type_enum == ObservationsType.from_csv:
        observation_config = ObservationsFromCSVConfig(
            type=obs_type_enum,
            df=pd.read_csv(observations_source_path, **csv_reader_options),
        )
    else:
        observation_config = ObservationsFromSimulationConfig(
            type=obs_type_enum,
            error_percent=observations_error_percent,
            locations=observation_locations,
            measure_resolution=measure_resolution,
            ds_reference_path=observations_source_path,
        )

    config = AssimilatorConfig(
        size_ensemble=size_ensemble,
        ensemble_spread=initial_ensemble_spread,
        initial_mass_multiplicator=initial_mass_multiplicator,
        num_particles_total=num_parts,
        grid_coords=RectGridCoords(
            x1=assimilation_domain_coords[0],
            y1=assimilation_domain_coords[1],
            x2=assimilation_domain_coords[2],
            y2=assimilation_domain_coords[3],
            spacing_x=(assimilation_domain_coords[2] - assimilation_domain_coords[0])
            / assimilation_grid_size[0],
            spacing_y=(assimilation_domain_coords[3] - assimilation_domain_coords[1])
            / assimilation_grid_size[1],
            max_lon_id=assimilation_grid_size[0],
            max_lat_id=assimilation_grid_size[1],
        ),
        radius_observation=radius_observation,
        graph_plot_period=metrics_plot_period,
        max_time=t_end + 1,
        t_start=t_start,
        t_end=t_end,
        reinit_spreading=reinit_spreading,
        observations=observation_config,
        cells_area=cells_area,
        verbose=verbose,
    )

    init_data(datapaths, config)

    start_simulation(datapaths, config)
