import numpy as np
import pandas as pd

from src.types import AssimilatorConfig, AssimilatorDataPaths
from src.assimilation.density_computations import compute_particle_ids_for_areas
from src.assimilation.density_computations_ensemble import (
    compute_ensemble_densities_over_parts,
)
from src.assimilation.localization import (
    compute_indices_circle,
    create_localization_matrix,
)
from src.io.plotting import gen_cov_map


def reintroduce_error(densities_ensemble, reinit_spreading: float, t_observation):
    for e in range(densities_ensemble.shape[0]):
        densities_ensemble[e, :, :, t_observation] += (
            np.random.randn() * reinit_spreading
        )


def compute_covariances_for_point(
    densities_ensemble, avgs_densities, lon_obs_id, lat_obs_id, t_observation
):
    return (
        np.sum(
            (
                densities_ensemble[:, lon_obs_id, lat_obs_id, t_observation]
                - avgs_densities[lon_obs_id, lat_obs_id],
            )
            * np.moveaxis(
                densities_ensemble[:, :, :, t_observation] - avgs_densities[:, :],
                0,
                -1,
            ),
            axis=2,
        )
        / (densities_ensemble.shape[0] - 1)
    )


def compute_covariances(
    t_observation: int,
    avgs_densities: np.ndarray,
    densities_ensemble: np.ndarray,
    observations: pd.DataFrame,
    localization_matrix: np.ndarray,
    metrics_dir_path: str,
    config: AssimilatorConfig,
):
    n, p = avgs_densities.shape
    cov = np.zeros((n, p, n, p))
    modifiedIndices = []

    for observation in observations.itertuples():
        lon_id = observation.lon_id
        lat_id = observation.lat_id

        if config.verbose:
            print("Computing covariances relevant for observation at", lon_id, lat_id)

        indices = compute_indices_circle(
            (lon_id, lat_id), config.radius_observation, n, p
        )
        modifiedIndices = list(set.union(set(modifiedIndices), set(indices)))

        cov[:, :, lon_id, lat_id] = (
            compute_covariances_for_point(
                densities_ensemble,
                avgs_densities,
                lon_id,
                lat_id,
                t_observation,
            )
            * localization_matrix[:, :, lon_id, lat_id]
        )

        if (t_observation - config.t_start) % config.graph_plot_period == 0:
            cov_mat_for_plot = np.zeros((n, p))
            for (x, y) in indices:
                cov_mat_for_plot[x, y] = cov[x, y, lon_id, lat_id]
            gen_cov_map(
                cov_mat_for_plot,
                "cov_mat_t"
                + str(t_observation)
                + "_"
                + str(lon_id)
                + "_"
                + str(lat_id),
                lon_id,
                lat_id,
                n,
                p,
                metrics_dir_path,
            )

    return cov, modifiedIndices


def compute_partial_kalman_gain(cov: np.ndarray, observations: pd.DataFrame):
    # A = (H * (C o P) * H.T + R) ** -1 * Differences
    B = np.zeros((len(observations), len(observations)))
    for i in range(len(observations)):
        for j in range(len(observations)):
            lon1, lat1 = observations[["lon_id", "lat_id"]].iloc[i]
            lon2, lat2 = observations[["lon_id", "lat_id"]].iloc[j]
            B[i, j] = cov[lon1, lat1, lon2, lat2]

    R = np.diag([(observation.variance) for observation in observations.itertuples()])

    B += R

    return np.linalg.inv(B)


def compute_corrections(
    t_observation: int,
    densities_ensemble: np.ndarray,
    cov: np.ndarray,
    observations: pd.DataFrame,
    Binv: np.ndarray,
    modifiedIndices: list,
    verbose: bool,
) -> np.ndarray:
    # Per ensemble correction
    corrections = np.zeros(
        (
            densities_ensemble.shape[0],
            densities_ensemble.shape[1],
            densities_ensemble.shape[2],
        )
    )

    obs_lon_ids = np.array(observations.lon_id)
    obs_lat_ids = np.array(observations.lat_id)
    obs_values = np.array(observations.value)

    for e in range(densities_ensemble.shape[0]):
        to_correct = np.array(
            [
                obs_values[i]
                - densities_ensemble[e, obs_lon_ids[i], obs_lat_ids[i], t_observation]
                for i in range(len(obs_values))
            ]
        )
        if verbose:
            print(
                *[
                    (
                        f"Observation {obs_values[i]} and prediction {densities_ensemble[e, obs_lon_ids[i], obs_lat_ids[i], t_observation]} gives value to correct {to_correct[i]}"
                    )
                    for i in range(len(obs_values))
                ]
            )
        A = np.dot(Binv, to_correct)

        for (x, y) in modifiedIndices:
            corrections[e, x, y] = sum(
                [
                    cov[x, y, obs_lon_ids[i], obs_lat_ids[i]] * A[i]
                    for i in range(len(obs_values))
                ]
            )

    return corrections


def update_weights(
    t_observation,
    weights,
    densities_ensemble,
    densities_ensemble_predicted,
    modifiedIndices,
    particleIdsForAreas,
):
    for xy in modifiedIndices:
        x, y = xy
        particles = particleIdsForAreas[x][y]

        totalWeight_predicted = densities_ensemble_predicted[:, x, y]
        totalWeight_corrected = densities_ensemble[:, x, y, t_observation]
        weights_predicted = np.moveaxis(weights[:, particles], 0, 1)
        weights_corrected = weights_predicted * (
            totalWeight_corrected / totalWeight_predicted
        )
        weights[:, particles] = np.moveaxis(weights_corrected, 0, 1)


def assimilate(
    t_observation: int,
    densities_ensemble: np.ndarray,
    observations: pd.DataFrame,
    weights: np.ndarray,
    parts_lon: np.ndarray,
    parts_lat: np.ndarray,
    config: AssimilatorConfig,
    datapaths: AssimilatorDataPaths,
):
    # prediction at given instant
    avgs_densities = np.average(densities_ensemble[:, :, :, t_observation], axis=0)
    densities_ensemble_predicted = densities_ensemble[:, :, :, t_observation].copy()

    if config.verbose:
        print("Computing localization matrix")
    localization_matrix = create_localization_matrix(
        config.grid_coords,
        observations,
        config.radius_observation,
    )

    if config.verbose:
        print("Computing particleIdsForAreas")
    particleIdsForAreas = compute_particle_ids_for_areas(
        parts_lon, parts_lat, t_observation, config.grid_coords
    )

    if t_observation != 0 and len(observations) != 0:
        if config.verbose:
            print("Introducing model density errors")
        reintroduce_error(densities_ensemble, config.reinit_spreading, t_observation)

    if config.verbose:
        print("Computing covariances")
    cov, modifiedIndices = compute_covariances(
        t_observation,
        avgs_densities,
        densities_ensemble,
        observations,
        localization_matrix,
        datapaths.metrics_dir,
        config,
    )

    if config.verbose:
        print("Computing partial Kalman Gain")
    Binv = compute_partial_kalman_gain(cov, observations)

    if config.verbose:
        print("Computing corrections")
    corrections = compute_corrections(
        t_observation,
        densities_ensemble,
        cov,
        observations,
        Binv,
        modifiedIndices,
        config.verbose,
    )
    if config.verbose:
        print("Maximum of corrections is ", corrections.max())
    if np.isnan(corrections.max()):
        return False
    densities_ensemble[:, :, :, t_observation] += corrections

    if config.verbose:
        print("Updating weights")
    update_weights(
        t_observation,
        weights,
        densities_ensemble,
        densities_ensemble_predicted,
        modifiedIndices,
        particleIdsForAreas,
    )

    if config.verbose:
        print("Recomputing densities for next day")
    particleIdsForAreas = compute_particle_ids_for_areas(
        parts_lon, parts_lat, t_observation + 1, config.grid_coords
    )
    compute_ensemble_densities_over_parts(
        particleIdsForAreas,
        densities_ensemble,
        weights,
        config.grid_coords,
        config.cells_area,
        t_observation + 1,
    )

    return True
