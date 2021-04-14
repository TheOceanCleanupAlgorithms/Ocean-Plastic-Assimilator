from typing import Optional
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4 as nc

from src.io.file_utils import create_folder
from src.io.CSV_Logger import CSV_Logger
from src.types import ObservationsType, RectGridCoords


class Metrics:
    def __init__(
        self,
        output_dir_path: str,
        densities_ensemble: np.ndarray,
        simulation_duration: int,
        observation_type: ObservationsType,
        grid_coords: RectGridCoords,
    ):
        self.output_dir_path = output_dir_path

        self.csv_logger = CSV_Logger(f"{output_dir_path}log")

        self.max_densities_error = 0
        self.min_densities_error = 0
        self.observation_type = observation_type
        self.grid_coords = grid_coords

        self.weights_means = []
        self.weights_sum = []
        self.weights_ref_sum = []
        self.densities_err_means = []
        self.densities_err_stddev = []
        self.densities_rmse = []

        self.avgs_densities_complete_original = np.average(
            densities_ensemble[:, :, :, np.arange(simulation_duration)], axis=0
        )

        create_folder(output_dir_path)

    def log_metrics(
        self,
        densities_ensemble: np.ndarray,
        densities_ref: Optional[np.ndarray],
        weights_ensemble: np.ndarray,
        parts_lon: np.ndarray,
        parts_lat: np.ndarray,
        t: int,
        parts_original_path: str,
    ):
        avgs_densities = np.average(densities_ensemble[:, :, :, t], axis=0)

        parts_lon_lats = np.column_stack((parts_lon[:, t], parts_lat[:, t]))
        valid_lon_lats = (
            (self.grid_coords.x1 <= parts_lon_lats[:, 0])
            & (self.grid_coords.x2 > parts_lon_lats[:, 0])
            & (self.grid_coords.y1 <= parts_lon_lats[:, 1])
            & (self.grid_coords.y2 > parts_lon_lats[:, 1])
        )

        avg_weights = np.average(weights_ensemble[:, valid_lon_lats], axis=0)

        if parts_original_path is not None:
            ds_ref = nc.Dataset(parts_original_path, "r")
            parts_ref_lon = ds_ref["lon"]
            parts_ref_lat = ds_ref["lat"]
            parts_ref_lon_lats = np.column_stack(
                (parts_ref_lon[:, t], parts_ref_lat[:, t])
            )
            valid_ref_lon_lats = (
                (self.grid_coords.x1 <= parts_ref_lon_lats[:, 0])
                & (self.grid_coords.x2 > parts_ref_lon_lats[:, 0])
                & (self.grid_coords.y1 <= parts_ref_lon_lats[:, 1])
                & (self.grid_coords.y2 > parts_ref_lon_lats[:, 1])
            )
            weights_ref_sum = np.sum(ds_ref["weight"][valid_ref_lon_lats])
        else:
            weights_ref_sum = 1

        densities_difference = (
            avgs_densities - densities_ref[:, :, t]
            if densities_ref is not None
            else avgs_densities - self.avgs_densities_complete_original[:, :, t]
        )

        self.max_densities_error = max(
            self.max_densities_error, densities_difference.max()
        )
        self.min_densities_error = min(
            self.min_densities_error, densities_difference.min()
        )
        current_rmse = np.math.sqrt(np.average(densities_difference ** 2))

        self.csv_logger.log("t", t + 1)
        self.csv_logger.log("weights_mins", np.min(avg_weights))
        self.csv_logger.log("weights_maxs", np.max(avg_weights))
        self.csv_logger.log("weights_means", np.mean(avg_weights))
        self.csv_logger.log("weights_sum", np.sum(avg_weights))
        self.csv_logger.log("weights_medians", np.median(avg_weights))
        self.csv_logger.log("weights_stddev", np.std(avg_weights))
        self.csv_logger.log("weights_ref_sum", weights_ref_sum)
        self.csv_logger.log("densities_err_mins", np.min(densities_difference))
        self.csv_logger.log("densities_err_maxs", np.max(densities_difference))
        self.csv_logger.log("densities_err_means", np.mean(densities_difference))
        self.csv_logger.log("densities_err_medians", np.median(densities_difference))
        self.csv_logger.log("densities_err_stddev", np.std(densities_difference))
        self.csv_logger.log("densities_rmse", current_rmse)
        self.csv_logger.flush()

        self.weights_means.append(np.mean(avg_weights))
        self.weights_sum.append(np.sum(avg_weights))
        self.weights_ref_sum.append(weights_ref_sum)
        self.densities_err_means.append(np.mean(densities_difference))
        self.densities_err_stddev.append(np.std(densities_difference))
        self.densities_rmse.append(current_rmse)

    def plot_metrics(
        self,
        densities_ensemble: np.ndarray,
        densities_ref: Optional[np.ndarray],
        weights_ensemble: np.ndarray,
        t: int,
    ):
        if self.observation_type == ObservationsType.from_simulation:
            self._plot_metrics_from_sim(
                densities_ensemble=densities_ensemble,
                densities_ref=densities_ref,
                weights_ensemble=weights_ensemble,
                t=t,
            )

        if self.observation_type == ObservationsType.from_csv:
            self._plot_metrics_from_csv(
                densities_ensemble=densities_ensemble,
                weights_ensemble=weights_ensemble,
                t=t,
            )

    def _plot_metrics_from_sim(
        self,
        densities_ensemble: np.ndarray,
        densities_ref: np.ndarray,
        weights_ensemble: np.ndarray,
        t: int,
    ):
        # Init plot
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        # Compute fields
        avgs_densities = np.average(densities_ensemble[:, :, :, t], axis=0)
        avg_weights = np.average(weights_ensemble, axis=0)

        densities_difference_to_ref = avgs_densities - densities_ref[:, :, t]
        densities_difference_to_origin = (
            avgs_densities - self.avgs_densities_complete_original[:, :, t]
        )

        # Draw fields
        sns.heatmap(
            densities_difference_to_origin.transpose(), robust=True, ax=axs[0, 0]
        ).set_facecolor("green")

        sns.heatmap(
            densities_difference_to_ref.transpose(),
            robust=True,
            vmin=self.min_densities_error,
            vmax=self.max_densities_error,
            center=0,
            ax=axs[0, 1],
        ).set_facecolor("green")

        sns.distplot(
            avg_weights[(0 < avg_weights[:]) & (avg_weights[:] < 3)],
            ax=axs[1, 0],
        )

        weights_means_df = pd.DataFrame(
            {"Forecast Mass": self.weights_sum, "Reference Mass": self.weights_ref_sum}
        )
        sns.lineplot(data=weights_means_df, ax=axs[1, 1])

        densities_err_means_df = pd.DataFrame(
            {"Densities error mean": self.densities_err_means}
        )
        densities_err_means_df["Objective"] = 0
        sns.lineplot(data=densities_err_means_df, ax=axs[0, 2])

        cfrms_df = pd.DataFrame({"Concentration Field Error RMS": self.densities_rmse})
        cfrms_df["Objective"] = 0
        sns.lineplot(data=cfrms_df, ax=axs[1, 2])

        self.csv_logger.export_csv()
        fig.savefig(f"{self.output_dir_path}metrics_{t}.png")
        plt.close("all")

    def _plot_metrics_from_csv(
        self,
        densities_ensemble: np.ndarray,
        weights_ensemble: np.ndarray,
        t: int,
    ):
        # Init plot
        fig, axs = plt.subplots(ncols=3, figsize=(35, 8))

        avgs_densities = np.average(densities_ensemble[:, :, :, t], axis=0)

        densities_difference_to_origin = (
            avgs_densities - self.avgs_densities_complete_original[:, :, t]
        )

        sns.set_context("talk")

        # Draw the field of differences to what would be if there was no assimilation
        sns.heatmap(
            densities_difference_to_origin.transpose(), robust=True, ax=axs[0], center=0
        )
        axs[0].set_title("Differences to original dispersion")

        # Draw the forecast mass of particles
        weights_means_df = pd.DataFrame({"Forecast Mass": self.weights_means})
        sns.lineplot(data=weights_means_df, ax=axs[1])
        axs[1].set_title(f"Forecast mass evolution. Current = {self.weights_means[-1]}")

        # Draw the current field
        sns.heatmap(avgs_densities.transpose(), robust=True, ax=axs[2])
        axs[2].set_title("Current concentrations")

        self.csv_logger.export_csv()
        fig.savefig(f"{self.output_dir_path}metrics_{t}.png")
        plt.close("all")
