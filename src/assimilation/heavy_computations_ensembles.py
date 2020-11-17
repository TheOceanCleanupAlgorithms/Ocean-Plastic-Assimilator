import numpy as np
from math import floor
import netCDF4 as nc
import multiprocessing as mp
from typing import List, Dict, Tuple
from sim_vars import (
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    NB_PARTS,
    RESOLUTION,
    LONGITUDES,
    LATITUDES,
    NB_ENSEMBLES,
)

VERBOSE = False


def merge_dict_of_lists(dict_list: List[Dict[Tuple[int, int], List[int]]]):
    res_dict = dict_list[0].copy()

    for i in range(1, len(dict_list)):
        d = dict_list[i]
        keys = d.keys()
        for key in keys:
            try:
                res_dict[key] += d[key]
            except KeyError:
                res_dict[key] = d[key]

    return res_dict


def computeParticleIdsForAreasParallel_unit(args):
    ds_particles_path, t, beginPart, endPart, threadNumber = args

    ds_particles = nc.Dataset(ds_particles_path, "r+")

    particleIdsForAreas = dict()

    lons = ds_particles.variables["lon"][beginPart:endPart, t]
    lats = ds_particles.variables["lat"][beginPart:endPart, t]

    for i in range(endPart - beginPart):
        lon = lons[i]
        lat = lats[i]

        if MIN_LATITUDE <= lat < MAX_LATITUDE and MIN_LONGITUDE <= lon < MAX_LONGITUDE:
            lonId = floor((lon - MIN_LONGITUDE) / RESOLUTION)
            latId = floor((lat - MIN_LATITUDE) / RESOLUTION)
            try:
                particleIdsForAreas[(lonId, latId)].append(beginPart + i)
            except KeyError:
                particleIdsForAreas[(lonId, latId)] = [beginPart + i]

        if i % 100 == 0 and VERBOSE:
            print(
                "Thread number",
                threadNumber,
                "has computed",
                100 * i / (endPart - beginPart),
            )

    return particleIdsForAreas


def computeParticleIdsForAreasParallel(ds_particles_path, t, nbThreads=1):
    ds_particles = nc.Dataset(ds_particles_path, "r+")

    nbPart = ds_particles["id"].shape[0]

    partsPerThread = int(nbPart / nbThreads)

    pool = mp.Pool(processes=nbThreads)

    args_p = [
        (ds_particles_path, t, i * partsPerThread, (i + 1) * partsPerThread, i)
        for i in range(nbThreads)
    ]

    results = [
        pool.apply_async(computeParticleIdsForAreasParallel_unit, args=([args_p[i]]))
        for i in range(nbThreads)
    ]

    results_dicts = []

    for i in range(nbThreads):
        results_dicts.append(results[i].get())

    return merge_dict_of_lists(results_dicts)


def computeEnsembleDensitiesOverTime(parts_lon, parts_lat, all_densities, weights, T):
    lons = parts_lon[:, T]
    lats = parts_lat[:, T]

    densities = np.zeros((LONGITUDES, LATITUDES, len(T), NB_ENSEMBLES))

    lonIdsForAllParts = np.floor((lons - MIN_LONGITUDE) / RESOLUTION).astype(
        int
    )  # ... TODO : COULD BE COMPUTED ONCE AND FOR ALL AT THE BEGINNING AND STORED IN A GLOBAL VAR
    latIdsForAllParts = np.floor((lats - MIN_LATITUDE) / RESOLUTION).astype(int)

    for i in range(NB_PARTS):
        lonIds = lonIdsForAllParts[i, :]
        latIds = latIdsForAllParts[i, :]
        lonIdsLatIdsT = np.array([lonIds, latIds, np.arange(len(T))]).T
        lonIdsLatIdsTFiltered = lonIdsLatIdsT[
            (lonIdsLatIdsT[:, 0] < LONGITUDES) & (lonIdsLatIdsT[:, 1] < LATITUDES)
        ]
        weight = weights[:, i]
        densities[
            lonIdsLatIdsTFiltered[:, 0],
            lonIdsLatIdsTFiltered[:, 1],
            lonIdsLatIdsTFiltered[:, 2],
            :,
        ] += weight

        if i % 100 == 0 and VERBOSE:
            print("Computing densities is done at ", 100 * i / (NB_PARTS), "%")

    all_densities[:, :, :, T] = np.moveaxis(densities, -1, 0)


def computeEnsembleDensitiesOverParts(partIdsForArea, all_densities, weights, t):
    densities = np.zeros((LONGITUDES, LATITUDES, NB_ENSEMBLES))

    for x in range(LONGITUDES):
        for y in range(LATITUDES):
            try:
                particleIds = partIdsForArea[(x, y)]
                weight = np.sum(weights[:, particleIds], axis=1)
                densities[x, y, :] += weight
            except KeyError:
                pass

    all_densities[:, :, :, t] = np.moveaxis(densities, -1, 0)
