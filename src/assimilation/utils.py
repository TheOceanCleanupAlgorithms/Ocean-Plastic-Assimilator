from typing import Tuple
import numpy as np
from sim_vars import LONGITUDES, LATITUDES, NB_ENSEMBLES


def computeIndicesCircle(center: Tuple[int, int], radius: int):
    xc, yc = center
    indices = []

    for x in range(max(xc - radius, 0), min(xc + radius + 1, LONGITUDES)):
        for y in range(max(yc - radius, 0), min(yc + radius + 1, LATITUDES)):
            if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) <= radius:
                indices.append((x, y))
    return indices


def computeCovariancesForPoint(
    densities_ensembles, avgs_densities, lon_obs_id, lat_obs_id, t_observation
):
    return (
        np.sum(
            (
                densities_ensembles[:, lon_obs_id, lat_obs_id, t_observation]
                - avgs_densities[lon_obs_id, lat_obs_id],
            )
            * np.moveaxis(
                densities_ensembles[:, :, :, t_observation] - avgs_densities[:, :],
                0,
                -1,
            ),
            axis=2,
        )
        / (NB_ENSEMBLES - 1)
    )
