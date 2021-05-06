from typing import List
from numba.core.decorators import njit
import numba.typed as nbt
import numpy as np

from src.types import RectGridCoords


def compute_ensemble_densities_over_time(
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    all_densities: np.ndarray,
    weights: np.ndarray,
    cells_area: np.ndarray,
    T: List,
    grid_coords: RectGridCoords,
):
    lons = parts_lon[:, T]
    lats = parts_lat[:, T]
    nbParts = parts_lon.shape[0]

    lon_ids_for_all_parts = np.floor(
        (lons - grid_coords.x1) / grid_coords.spacing_x
    ).astype(int)
    lat_ids_for_all_parts = np.floor(
        (lats - grid_coords.y1) / grid_coords.spacing_y
    ).astype(int)

    densities = llvm_compute_ensemble_densities_over_time(
        lon_ids_for_all_parts,
        lat_ids_for_all_parts,
        weights,
        cells_area,
        nbParts,
        grid_coords.max_lon_id,
        grid_coords.max_lat_id,
        len(T),
        weights.shape[0],
    )

    all_densities[:, :, :, T] = np.moveaxis(densities, -1, 0)


@njit
def llvm_compute_ensemble_densities_over_time(
    lon_ids_for_all_parts, lat_ids_for_all_parts, weights, cells_area, nbParts, n, p, T, size_e
):
    densities = np.zeros((n, p, T, size_e))

    for i in range(nbParts):
        for t in range(T):
            lonId = lon_ids_for_all_parts[i, t]
            latId = lat_ids_for_all_parts[i, t]

            if lonId >= 0 and lonId < n and latId >= 0 and latId < p:
                for e in range(size_e):
                    densities[lonId, latId, t, e] += weights[e, i] / cells_area[lonId, latId]

    return densities


def compute_ensemble_densities_over_parts(
    partIdsForArea: nbt.List,
    all_densities: np.ndarray,
    weights: np.ndarray,
    grid_coords: RectGridCoords,
    cells_area: np.ndarray,
    t: int,
):
    densities = np.zeros(
        (grid_coords.max_lon_id, grid_coords.max_lat_id, weights.shape[0])
    )

    for x in range(grid_coords.max_lon_id):
        for y in range(grid_coords.max_lat_id):
            try:
                particleIds = partIdsForArea[x][y]
                weight = np.sum(weights[:, particleIds], axis=1)
                densities[x, y, :] += weight / cells_area[x, y]
            except KeyError:
                pass

    all_densities[:, :, :, t] = np.moveaxis(densities, -1, 0)
