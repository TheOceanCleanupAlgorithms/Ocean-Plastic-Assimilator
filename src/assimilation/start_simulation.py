import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import netCDF4 as nc
import os
import seaborn as sns
import pandas as pd
from src.assimilation.assimilate import assimilate

from sim_vars import (
    LATITUDES,
    LONGITUDES,
    TIMES,
    observation_locations,
    radius_obs,
    t_start_iter,
    t_end_iter,
    ds_parts_ensembles_path,
    ds_densities_ref_path,
    ds_densities_ensembles_path,
    output_dir_path,
    GRAPH_PLOT_PERIOD,
)
from src.assimilation.pandas_logger import Logger


def start_simulation():
    matplotlib.use("agg")  # Necessary for multithreading, somehow

    try:
        print("Creating output folder...")
        os.mkdir(output_dir_path)
    except OSError:
        print(
            "There is already an output folder! Please remove and backup it before starting "
        )
        exit()

    logger = Logger(output_dir_path + "log")

    print("Retrieving necessary datasets")
    ds_parts_ensembles = nc.Dataset(ds_parts_ensembles_path, "r")
    weights = ds_parts_ensembles["weight"][:, :]
    parts_lon = ds_parts_ensembles["lon"][:, :]
    parts_lat = ds_parts_ensembles["lat"][:, :]
    ds_parts_ensembles.close()

    ds_densities_ref = nc.Dataset(ds_densities_ref_path)

    print("Computing first ensembles mean")
    ds_densities_ensembles = nc.Dataset(ds_densities_ensembles_path, "r")
    densities_ensembles = ds_densities_ensembles["density"][:, :, :, :]
    avgs_densities_complete_original = np.average(
        densities_ensembles[:, :, :, np.arange(TIMES)], axis=0
    )
    ds_densities_ensembles.close()

    print("Getting reference data")
    densities_ref = ds_densities_ref["density"][:, :, :]

    # =================================================== INITIAL CACHING FOR STUF THAT CAN BE ================================

    print("Computing localization matrix")

    localizationMatrix = np.ones((LONGITUDES, LATITUDES, LONGITUDES, LATITUDES))

    for lon_obs_id, lat_obs_id in observation_locations:
        loc_factor = lambda x1, y1, x2, y2: max(
            1 - np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / radius_obs, 0
        )
        localizationMatrix[:, :, lon_obs_id, lat_obs_id] = np.array(
            [
                [loc_factor(x, y, lon_obs_id, lat_obs_id) for y in range(LATITUDES)]
                for x in range(LONGITUDES)
            ]
        )

    # =================================================== INITIAL METRICS ========================================================

    print("Generating initial metrics")
    sns.heatmap(
        avgs_densities_complete_original[:, :, t_start_iter]
    ).get_figure().savefig(f"{output_dir_path}_densities_at_beginning")

    avg_weights = np.average(weights, axis=0)
    densities_difference_to_ref = (
        avgs_densities_complete_original[:, :, t_start_iter]
        - ds_densities_ref["density"][:, :, t_start_iter]
    )
    max_densities_error = densities_difference_to_ref.max()
    min_densities_error = densities_difference_to_ref.min()

    # Metrics data
    logger.log("weights_mins", np.min(avg_weights))
    logger.log("weights_maxs", np.max(avg_weights))
    logger.log("weights_means", np.mean(avg_weights))
    logger.log("weights_medians", np.median(avg_weights))
    logger.log("weights_stddev", np.std(avg_weights))
    logger.log("densities_err_mins", np.min(densities_difference_to_ref))
    logger.log("densities_err_maxs", np.max(densities_difference_to_ref))
    logger.log("densities_err_means", np.mean(densities_difference_to_ref))
    logger.log("densities_err_medians", np.median(densities_difference_to_ref))
    logger.log("densities_err_stddev", np.std(densities_difference_to_ref))
    logger.flush()

    # Initialize subplots
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(20, 10))

    weights_means = []
    densities_err_means = []
    densities_err_stddev = []
    cfrms = []

    # =================================================== ITERATIONS ========================================================

    try:
        print("Start iterating")
        for t in range(t_start_iter, t_end_iter):
            # ------------------- STOP IF ASKED -------------------
            if os.path.isfile("stop"):
                os.remove("stop")
                print(
                    "Stop file has been found at the source of code. Stopping simulation now"
                )
                break

            print("=================================================")
            print(
                "Start iteration", t - t_start_iter + 1, "on", t_end_iter - t_start_iter
            )

            # ------------------- ASSIMILATE ----------------------

            assimilate(
                t,
                densities_ensembles,
                densities_ref,
                weights,
                parts_lon,
                parts_lat,
                localizationMatrix,
            )

            # -------------------- METRICS COMPUTING ------------------------

            avgs_densities = np.average(densities_ensembles[:, :, :, t + 1], axis=0)
            avg_weights = np.average(weights, axis=0)

            densities_difference_to_ref = avgs_densities - densities_ref[:, :, t + 1]

            max_densities_error = max(
                max_densities_error, densities_difference_to_ref.max()
            )
            min_densities_error = min(
                min_densities_error, densities_difference_to_ref.min()
            )
            current_cfrms = np.math.sqrt(np.sum(densities_difference_to_ref ** 2))

            logger.log("t", t + 1)
            logger.log("weights_mins", np.min(avg_weights))
            logger.log("weights_maxs", np.max(avg_weights))
            logger.log("weights_means", np.mean(avg_weights))
            logger.log("weights_medians", np.median(avg_weights))
            logger.log("weights_stddev", np.std(avg_weights))
            logger.log("densities_err_mins", np.min(densities_difference_to_ref))
            logger.log("densities_err_maxs", np.max(densities_difference_to_ref))
            logger.log("densities_err_means", np.mean(densities_difference_to_ref))
            logger.log("densities_err_medians", np.median(densities_difference_to_ref))
            logger.log("densities_err_stddev", np.std(densities_difference_to_ref))
            logger.log("CFRMS", current_cfrms)
            logger.flush()

            weights_means.append(np.mean(avg_weights))
            densities_err_means.append(np.mean(densities_difference_to_ref))
            densities_err_stddev.append(np.std(densities_difference_to_ref))
            cfrms.append(current_cfrms)

            # -------------------- PERIODIC PLOTTING AND EXPORTS -----------------------

            if (t - t_start_iter) % GRAPH_PLOT_PERIOD == 0:
                fig.clear()
                axs = fig.subplots(2, 3)

                print("Generating heatmaps and distributions")
                diff_densities_origin = (
                    avgs_densities - avgs_densities_complete_original[:, :, t + 1]
                )

                sns.heatmap(
                    diff_densities_origin.transpose(), robust=True, ax=axs[0, 0]
                )

                sns.heatmap(
                    densities_difference_to_ref.transpose(),
                    robust=True,
                    vmin=min_densities_error,
                    vmax=max_densities_error,
                    center=0,
                    ax=axs[0, 1],
                )

                sns.distplot(
                    avg_weights[(0 < avg_weights[:]) & (avg_weights[:] < 3)],
                    ax=axs[1, 0],
                )

                weights_means_df = pd.DataFrame({"Forecast Mass": weights_means})
                weights_means_df["Reference Mass"] = 1
                sns.lineplot(data=weights_means_df, ax=axs[1, 1])

                densities_err_means_df = pd.DataFrame(
                    {"Densities error mean": densities_err_means}
                )
                densities_err_means_df["Objective"] = 0
                sns.lineplot(data=densities_err_means_df, ax=axs[0, 2])

                cfrms_df = pd.DataFrame({"Concentration Field Error RMS": cfrms})
                cfrms_df["Objective"] = 0
                sns.lineplot(data=cfrms_df, ax=axs[1, 2])

                logger.export_csv()
                fig.savefig(f"{output_dir_path}metrics_{t}.png")
                plt.close("all")

    except KeyboardInterrupt:
        print("Asked for interruption!")

    print("Saving everything now...")

    ds_densities_ensembles = nc.Dataset(ds_densities_ensembles_path, "r+")
    ds_densities_ensembles["density"][:, :, :, :] = densities_ensembles
    ds_densities_ensembles.close()

    ds_parts_ensembles = nc.Dataset(ds_parts_ensembles_path, "r+")
    ds_parts_ensembles["weight"][:, :] = weights
    ds_parts_ensembles.close()

    plt.close("all")
    logger.export_csv()


if __name__ == "__main__":
    start_simulation()
