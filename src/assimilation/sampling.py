import pandas as pd
import numpy as np


def sample_observations(
    densities_ref: np.ndarray,
    observation_locations: list,
    observation_error_percent: float,
    measure_resolution: float,
    t: int,
) -> pd.DataFrame:
    lon_ids = [loc[0] for loc in observation_locations]
    lat_ids = [loc[1] for loc in observation_locations]

    # Simulate observation errors based on the desired percentage error
    observation_errors = [
        (
            observation_error_percent
            * densities_ref[lon_id, lat_id, t]
            * np.random.randn()
        )
        for lon_id, lat_id in zip(lon_ids, lat_ids)
    ]

    # Define the observation values based on the desired percentage error and reference dispersion
    observation_values = [
        max(
            densities_ref[lon_id, lat_id, t] + observation_error,
            0,
        )
        for lon_id, lat_id, observation_error in zip(
            lon_ids, lat_ids, observation_errors
        )
    ]

    # Define observation variance based on desired percentage error and measure resolution.
    # We assume there's no covariance between measurements.
    observation_variances = [
        (observation_error_percent * observation_value) ** 2 + measure_resolution ** 2
        for observation_value in observation_values
    ]

    df_observations = pd.DataFrame(
        {
            "lon_id": lon_ids,
            "lat_id": lat_ids,
            "value": observation_values,
            "variance": observation_variances,
        }
    )

    return df_observations
