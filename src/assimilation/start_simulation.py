import matplotlib.pyplot as plt
import netCDF4 as nc
import pandas as pd
from src.io.Metrics import Metrics
from src.assimilation.sampling import sample_observations
from src.assimilation.assimilate import assimilate

from src.types import AssimilatorConfig, AssimilatorDataPaths, ObservationsType


def start_simulation(datapaths: AssimilatorDataPaths, config: AssimilatorConfig):
    if config.verbose:
        print("Opening necessary datasets")
    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "r")
    weights = ds_parts_ensembles["weight"][:, :]
    parts_lon = ds_parts_ensembles["lon"][:, :]
    parts_lat = ds_parts_ensembles["lat"][:, :]
    ds_parts_ensembles.close()

    ds_densities_ensemble = nc.Dataset(datapaths.ds_densities_ensemble, "r")
    densities_ensemble = ds_densities_ensemble["density"][:, :, :, :]
    ds_densities_ensemble.close()

    if config.observations.type == ObservationsType.from_simulation:
        if config.verbose:
            print("Getting reference data")
        ds_densities_ref = nc.Dataset(datapaths.ds_densities_ref)
        densities_ref = ds_densities_ref["density"][:, :, :]
    else:
        densities_ref = None

    # =================================================== INITIAL METRICS ========================================================
    if config.verbose:
        print("Generating initial metrics")

    metrics = Metrics(
        datapaths.metrics_dir,
        densities_ensemble,
        config.max_time,
        config.observations.type,
        config.grid_coords,
    )
    metrics.log_metrics(
        densities_ensemble,
        densities_ref,
        weights,
        parts_lon,
        parts_lat,
        config.t_start,
        parts_original_path=config.observations.ds_reference_path
        if config.observations.type == ObservationsType.from_simulation
        else None,
    )

    # =================================================== ITERATIONS ========================================================

    try:
        print("Start iterating")
        for t in range(config.t_start, config.t_end):
            print("=================================================")
            print(
                f"Start iteration {t - config.t_start + 1} / {config.t_end - config.t_start}"
            )

            if config.observations.type == ObservationsType.from_simulation:
                if config.verbose:
                    print("Sampling observations from reference simulation")
                observations = sample_observations(
                    densities_ref,
                    config.observations.locations,
                    config.observations.error_percent,
                    config.observations.measure_resolution,
                    t,
                )
            else:
                if config.verbose:
                    print("Retrieving observations from csv")
                df_observations = config.observations.df
                observations: pd.DataFrame = df_observations[
                    df_observations["time"] == t
                ]

            assimilate_res = assimilate(
                t,
                densities_ensemble,
                observations,
                weights,
                parts_lon,
                parts_lat,
                config,
                datapaths,
            )

            if not assimilate_res:
                break

            metrics.log_metrics(
                densities_ensemble,
                densities_ref,
                weights,
                parts_lon,
                parts_lat,
                t + 1,
                parts_original_path=config.observations.ds_reference_path
                if config.observations.type == ObservationsType.from_simulation
                else None,
            )
            if (t - config.t_start) % config.graph_plot_period == 0:
                if config.verbose:
                    print("Generating heatmaps and distributions")
                metrics.plot_metrics(densities_ensemble, densities_ref, weights, t + 1)

    except KeyboardInterrupt:
        print("Asked for interruption!")

    print("Saving everything now...\nDo not interrupt.")

    ds_densities_ensemble = nc.Dataset(datapaths.ds_densities_ensemble, "r+")
    ds_densities_ensemble["density"][:, :, :, :] = densities_ensemble
    ds_densities_ensemble.close()

    ds_parts_ensembles = nc.Dataset(datapaths.ds_parts_ensemble, "r+")
    ds_parts_ensembles["weight"][:, :] = weights
    ds_parts_ensembles.close()

    plt.close("all")
    metrics.csv_logger.export_csv()
