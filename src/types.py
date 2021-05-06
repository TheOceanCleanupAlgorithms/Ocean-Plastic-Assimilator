from dataclasses import dataclass
from typing import Tuple, Union, List
from enum import Enum

import pandas as pd
import numpy as np


@dataclass
class AssimilatorDataPaths:
    """Dataclass to store path towards different datasets clearly over all the program"""

    metrics_dir: str
    data_dir: str
    ds_parts_original: str
    ds_parts_ensemble: str
    ds_densities_ensemble: str
    ds_densities_ref: Union[str, None] = None


@dataclass
class RectGridCoords:
    x1: float
    y1: float
    x2: float
    y2: float
    spacing_x: float
    spacing_y: float
    max_lon_id: int
    max_lat_id: int


class ObservationsType(Enum):
    """Allows to handle different types of observation sources."""

    from_simulation = 0
    from_csv = 1


@dataclass
class ObservationsFromCSVConfig:

    type: ObservationsType.from_csv
    df: pd.DataFrame


@dataclass
class ObservationsFromSimulationConfig:

    type: ObservationsType.from_simulation
    error_percent: float
    locations: List[Tuple[int, int]]
    measure_resolution: float
    ds_reference_path: str


@dataclass
class AssimilatorConfig:
    """Dataclass to store Assimilator configuration variables"""

    size_ensemble: int
    ensemble_spread: float
    initial_mass_multiplicator: float
    num_particles_total: int
    grid_coords: RectGridCoords
    radius_observation: int
    graph_plot_period: int
    max_time: int
    t_start: int
    t_end: int
    reinit_spreading: float
    verbose: bool
    observations: Union[ObservationsFromCSVConfig, ObservationsFromSimulationConfig]
    cells_area: np.ndarray
