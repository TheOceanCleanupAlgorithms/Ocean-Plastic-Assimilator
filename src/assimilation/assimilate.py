import numpy as np

from src.assimilation.heavy_computations import computeParticleIdsForAreasParallel
from src.assimilation.heavy_computations_ensembles import (
    computeEnsembleDensitiesOverParts,
)
from src.assimilation.utils import (
    computeCovariancesForPoint,
    computeIndicesCircle,
)
from src.assimilation.plotting import gen_cov_map

from sim_vars import (
    REINIT_STD_DEV,
    observation_locations,
    radius_obs,
    t_start_iter,
    NB_ENSEMBLES,
    GRAPH_PLOT_PERIOD,
    PLOT_COV_MATRICES,
    observation_cov,
    measure_resolution,
)


def assimilate(
    t_observation,
    densities_ensembles,
    densities_ref,
    weights,
    parts_lon,
    parts_lat,
    localizationMatrix,
):
    # prediction at given instant
    avgs_densities = np.average(densities_ensembles[:, :, :, t_observation], axis=0)

    densities_ensembles_predicted = densities_ensembles[:, :, :, t_observation].copy()

    n, p = avgs_densities.shape

    print("Computing particleIdsForAreas")
    particleIdsForAreas = computeParticleIdsForAreasParallel(
        parts_lon, parts_lat, t_observation
    )

    modifiedIndices = []

    if t_observation != 0:
        print("Introducing model density errors")
        for e in range(NB_ENSEMBLES):
            densities_ensembles[e, :, :, t_observation] += (
                np.random.randn() * REINIT_STD_DEV
            )

    # ASSIMILATION FOR EACH POINT OF OBSERVATION
    # This is really the heart of this project.
    # Make sure every variable is well named.

    # cov = C o P where C is the localization correlation matrix, P is the correlation matrix, and o is the Hadamard product
    cov = np.zeros((n, p, n, p))
    measuredValues = dict()

    for lon_obs_id, lat_obs_id in observation_locations:
        print(
            "Computing covariances relevant for observation at", lon_obs_id, lat_obs_id
        )

        indices = computeIndicesCircle((lon_obs_id, lat_obs_id), radius_obs)
        modifiedIndices = list(set.union(set(modifiedIndices), set(indices)))
        observation_error = (
            observation_cov
            * densities_ref[lon_obs_id, lat_obs_id, t_observation]
            * np.random.randn()
        )
        measuredValues[(lon_obs_id, lat_obs_id)] = max(
            densities_ref[lon_obs_id, lat_obs_id, t_observation] + observation_error,
            0,
        )

        cov[:, :, lon_obs_id, lat_obs_id] = (
            computeCovariancesForPoint(
                densities_ensembles,
                avgs_densities,
                lon_obs_id,
                lat_obs_id,
                t_observation,
            )
            * localizationMatrix[:, :, lon_obs_id, lat_obs_id]
        )

        if (
            t_observation - t_start_iter
        ) % GRAPH_PLOT_PERIOD == 0 and PLOT_COV_MATRICES:
            cov_mat_for_plot = np.zeros((n, p))
            for (x, y) in indices:
                cov_mat_for_plot[x, y] = cov[x, y, lon_obs_id, lat_obs_id]
            gen_cov_map(
                cov_mat_for_plot,
                "cov_mat_t"
                + str(t_observation)
                + "_"
                + str(lon_obs_id)
                + "_"
                + str(lat_obs_id),
                lon_obs_id,
                lat_obs_id,
            )

    print("Computing and applying Kalman Gain")

    # A = (H * (C o P) * H.T + R) ** -1 * Differences
    B = np.zeros((len(observation_locations), len(observation_locations)))
    for i in range(len(observation_locations)):
        for j in range(len(observation_locations)):
            lon1, lat1 = observation_locations[i]
            lon2, lat2 = observation_locations[j]
            B[i, j] = cov[lon1, lat1, lon2, lat2]

    R = np.diag(
        [
            (observation_cov * measuredValues[(lon_obs_id, lat_obs_id)]) ** 2
            for lon_obs_id, lat_obs_id in observation_locations
        ]
    ) + np.diag([measure_resolution ** 2] * len(observation_locations))

    B += R

    Binv = np.linalg.inv(B)

    # Per ensemble correction
    for e in range(NB_ENSEMBLES):
        to_correct = np.array(
            [
                measuredValues[(lon_obs_id, lat_obs_id)]
                - densities_ensembles[e, lon_obs_id, lat_obs_id, t_observation]
                for lon_obs_id, lat_obs_id in observation_locations
            ]
        )
        A = np.dot(Binv, to_correct)

        corrections = np.zeros(
            (densities_ensembles.shape[1], densities_ensembles.shape[2])
        )

        for (x, y) in modifiedIndices:
            corrections[x, y] = sum(
                [
                    cov[x, y, observation_locations[i][0], observation_locations[i][1]]
                    * A[i]
                    for i in range(len(observation_locations))
                ]
            )

        densities_ensembles[e, :, :, t_observation] += corrections

    # END ASSIMILATION

    print("Updating weights")
    for xy in modifiedIndices:
        x, y = xy
        try:
            particles = particleIdsForAreas[(x, y)]
        except KeyError:
            continue
        else:
            totalWeight_predicted = densities_ensembles_predicted[:, x, y]
            totalWeight_corrected = densities_ensembles[:, x, y, t_observation]
            weights_predicted = np.moveaxis(weights[:, particles], 0, 1)
            weights_corrected = weights_predicted * (
                totalWeight_corrected / totalWeight_predicted
            )
            weights[:, particles] = np.moveaxis(weights_corrected, 0, 1)

    print("Recomputing densities for next day")
    particleIdsForAreas = computeParticleIdsForAreasParallel(
        parts_lon, parts_lat, t_observation + 1
    )
    computeEnsembleDensitiesOverParts(
        particleIdsForAreas,
        densities_ensembles,
        weights,
        t_observation + 1,
    )
