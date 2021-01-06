from typing import Tuple
import numpy as np
import pandas as pd

from src.types import RectGridCoords


def create_localization_matrix(
    grid_coords: RectGridCoords, observations: pd.DataFrame, radius_observation: int
):
    localization_matrix = np.ones(
        (
            grid_coords.max_lon_id,
            grid_coords.max_lat_id,
            grid_coords.max_lon_id,
            grid_coords.max_lat_id,
        )
    )

    for observation in observations[["lon_id", "lat_id"]].itertuples():

        def loc_factor(x1, y1, x2, y2):
            return max(
                1 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / radius_observation,
                0,
            )

        localization_matrix[:, :, observation.lon_id, observation.lat_id] = np.array(
            [
                [
                    loc_factor(x, y, observation.lon_id, observation.lat_id)
                    for y in range(grid_coords.max_lat_id)
                ]
                for x in range(grid_coords.max_lon_id)
            ]
        )

    return localization_matrix


def compute_indices_circle(
    center: Tuple[int, int], radius: int, max_lon_id: int, max_lat_id: int
):
    xc, yc = center
    indices = []

    for x in range(max(xc - radius, 0), min(xc + radius + 1, max_lon_id)):
        for y in range(max(yc - radius, 0), min(yc + radius + 1, max_lat_id)):
            if np.sqrt((x - xc) ** 2 + (y - yc) ** 2) <= radius:
                indices.append((x, y))
    return indices
