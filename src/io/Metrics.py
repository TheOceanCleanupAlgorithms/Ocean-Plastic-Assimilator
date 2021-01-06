from typing import Union
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from src.io.file_utils import create_folder
from src.io.CSV_Logger import CSV_Logger


class Metrics:
    def __init__(
        self,
        output_dir_path: str,
        densities_ensemble: np.ndarray,
        simulation_duration: int,
    ):
        self.output_dir_path = output_dir_path

        self.csv_logger = CSV_Logger(f"{output_dir_path}log")

        self.max_densities_error = 0
        self.min_densities_error = 0

        self.weights_means = []
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
        densities_ref: Union[np.ndarray, None],
        weights_ensemble: np.ndarray,
        t: int,
    ):
        avgs_densities = np.average(densities_ensemble[:, :, :, t], axis=0)
        avg_weights = np.average(weights_ensemble, axis=0)

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
        self.csv_logger.log("weights_medians", np.median(avg_weights))
        self.csv_logger.log("weights_stddev", np.std(avg_weights))
        self.csv_logger.log("densities_err_mins", np.min(densities_difference))
        self.csv_logger.log("densities_err_maxs", np.max(densities_difference))
        self.csv_logger.log("densities_err_means", np.mean(densities_difference))
        self.csv_logger.log("densities_err_medians", np.median(densities_difference))
        self.csv_logger.log("densities_err_stddev", np.std(densities_difference))
        self.csv_logger.log("densities_rmse", current_rmse)
        self.csv_logger.flush()

        self.weights_means.append(np.mean(avg_weights))
        self.densities_err_means.append(np.mean(densities_difference))
        self.densities_err_stddev.append(np.std(densities_difference))
        self.densities_rmse.append(current_rmse)

    def plot_metrics(
        self,
        densities_ensemble: np.ndarray,
        densities_ref: Union[np.ndarray, None],
        weights_ensemble: np.ndarray,
        t: int,
    ):
        # Init plot
        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

        # Compute fields
        avgs_densities = np.average(densities_ensemble[:, :, :, t], axis=0)
        avg_weights = np.average(weights_ensemble, axis=0)

        densities_to_plot = (
            avgs_densities - densities_ref[:, :, t]
            if densities_ref is not None
            else avgs_densities
        )
        densities_difference_to_origin = (
            avgs_densities - self.avgs_densities_complete_original[:, :, t]
        )

        # Draw fields
        sns.heatmap(
            densities_difference_to_origin.transpose(), robust=True, ax=axs[0, 0]
        )

        sns.heatmap(
            densities_to_plot.transpose(),
            robust=True,
            vmin=self.min_densities_error if densities_ref is not None else None,
            vmax=self.max_densities_error if densities_ref is not None else None,
            center=0,
            ax=axs[0, 1],
        )

        sns.distplot(
            avg_weights[(0 < avg_weights[:]) & (avg_weights[:] < 3)],
            ax=axs[1, 0],
        )

        weights_means_df = pd.DataFrame({"Forecast Mass": self.weights_means})
        weights_means_df["Reference Mass"] = 1
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
