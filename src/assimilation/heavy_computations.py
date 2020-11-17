import numpy as np
from math import floor
import netCDF4 as nc
import multiprocessing as mp
from sim_vars import (
    MAX_LATITUDE,
    MAX_LONGITUDE,
    MIN_LATITUDE,
    MIN_LONGITUDE,
    RESOLUTION,
    LONGITUDES,
    LATITUDES,
    TIMES,
    NB_PARTS,
)

VERBOSE = False


def computeParticleIdsForAreasParallel(parts_lon, parts_lat, t):
    particleIdsForAreas = dict()

    lons = parts_lon[:, t]
    lats = parts_lat[:, t]

    for i in range(NB_PARTS):
        lon = lons[i]
        lat = lats[i]

        if MIN_LATITUDE <= lat < MAX_LATITUDE and MIN_LONGITUDE <= lon < MAX_LONGITUDE:
            lonId = floor((lon - MIN_LONGITUDE) / RESOLUTION)
            latId = floor((lat - MIN_LATITUDE) / RESOLUTION)
            try:
                particleIdsForAreas[(lonId, latId)].append(i)
            except KeyError:
                particleIdsForAreas[(lonId, latId)] = [i]

        if VERBOSE:
            print("Particle ids for areas computed at ", 100 * i / NB_PARTS, "%")

    return particleIdsForAreas


def computeDensitiesParallel_unit(args):
    ds_in_path, T, beginPart, endPart, threadNumber = args

    ds_in = nc.Dataset(ds_in_path, "r")

    lons = ds_in.variables["lon"][beginPart:endPart, T]
    lats = ds_in.variables["lat"][beginPart:endPart, T]
    weights = ds_in.variables["weight"][beginPart:endPart]

    densities = np.zeros((LONGITUDES, LATITUDES, TIMES))

    lonIdsForAllParts = np.floor((lons - MIN_LONGITUDE) / RESOLUTION).astype(
        int
    )  # ... TODO : COULD BE COMPUTED ONCE AND FOR ALL AT THE BEGINNING AND STORED IN A GLOBAL VAR
    latIdsForAllParts = np.floor((lats - MIN_LATITUDE) / RESOLUTION).astype(int)

    for i in range(endPart - beginPart):
        lonIds = lonIdsForAllParts[i, T]
        latIds = latIdsForAllParts[i, T]
        lonIdsLatIdsT = np.array([lonIds, latIds, T]).T
        lonIdsLatIdsTFiltered = lonIdsLatIdsT[
            (lonIdsLatIdsT[:, 0] < LONGITUDES) & (lonIdsLatIdsT[:, 1] < LATITUDES)
        ]
        weight = weights[i]

        densities[
            lonIdsLatIdsTFiltered[:, 0],
            lonIdsLatIdsTFiltered[:, 1],
            lonIdsLatIdsTFiltered[:, 2],
        ] += weight

        if i % 100 == 0 and VERBOSE:
            print(
                "Thread number",
                threadNumber,
                "has computed",
                100 * i / (endPart - beginPart),
            )

    return densities


def computeDensitiesParallel(ds_in_path, ds_out_path, T, nbThreads=1):
    ds_in = nc.Dataset(ds_in_path, "r")
    nbPart = ds_in["id"].shape[0]
    ds_out = nc.Dataset(ds_out_path, "r+")

    if nbThreads > 1:
        partsPerThread = int(nbPart / nbThreads)

        pool = mp.Pool(processes=nbThreads)

        args_p = [
            (ds_in_path, T, i * partsPerThread, (i + 1) * partsPerThread, i)
            for i in range(nbThreads)
        ]

        results = [
            pool.apply_async(computeDensitiesParallel_unit, args=([args_p[i]]))
            for i in range(nbThreads)
        ]

        new_densities = np.zeros((LONGITUDES, LATITUDES, TIMES))

        for i in range(nbThreads):
            new_densities += results[i].get()

    else:
        new_densities = computeDensitiesParallel_unit((ds_in_path, T, 0, nbPart, -1))

    ds_out["density"][:, :, T] = new_densities[:, :, T]
