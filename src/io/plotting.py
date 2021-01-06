import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def gen_cov_map(
    dataset, name, x, y, max_lon_id, max_lat_id, metrics_dir_path, axes=None
):
    if axes is None:
        plt.figure(figsize=(12, 8))
    xlabels = np.full(max_lon_id, "")
    xlabels[x] = x
    ylabels = np.full(max_lat_id, "")
    ylabels[y] = y
    sns.set_context("paper")
    ax = sns.heatmap(
        dataset.transpose(),
        robust=True,
        center=dataset[x, y],
        xticklabels=False,
        yticklabels=False,
        cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0),
        ax=axes,
    )
    plt.xticks([x + 0.5], [str(x)], rotation=90)
    plt.yticks([y + 0.5], [str(y)])
    ax.get_figure().savefig(metrics_dir_path + name)
