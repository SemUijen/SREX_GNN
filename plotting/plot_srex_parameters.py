from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List

import torch


def plot_parameters_2D(
        x,
        y,
        col,
        axis_labels: Tuple[str, str],
        ax: Optional[plt.Axes] = None,
        cmap: Optional[Colormap] = None,
        lim_labels: Optional[bool] = False):
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not ax:
        _, ax = plt.subplots()

    if lim_labels:
        ax.scatter(x, y, cmap=cmap, c=col, alpha=0.9, vmin=0, vmax=1)
    else:
        ax.scatter(x, y, cmap=cmap, c=col, alpha=col, vmin=0, vmax=1)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])


def plot_parameters_3D(
        x,
        y,
        z,
        col,
        axis_labels: Tuple[str, str, str],
        ax: Optional[plt.Axes] = None,
        cmap: Optional[Colormap] = None,
        lim_labels: Optional[bool] = False):
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not ax:
        _, ax = plt.subplots()

    if lim_labels:
        clm = ax.scatter(x, y, z, c=col, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
    else:
        clm = ax.scatter(x[col > 0.5], y[col > 0.5], z[col > 0.5], c=col[col > 0.5], cmap=cmap, alpha=0.9, vmin=0, vmax=1)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    return clm


def plot_srex_parameters(
        parameter_array,
        fig: Optional[plt.figure] = None,
        cmap: Optional[Colormap] = None,
        title: Optional[str] = None,
        lim_labels: Optional[bool] = False,
        indices: Optional[List[int]] = None
) -> None:
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not fig:
        fig = plt.figure(figsize=(20, 12))

    gs = fig.add_gridspec(3, 2, width_ratios=(2 / 5, 3 / 5))

    if lim_labels:
        move = np.array([item[0] for item in indices])
        p1 = np.array([item[1] for item in indices])
        p2 = np.array([item[1] for item in indices])
        col = np.array(parameter_array)

    else:
        move = np.indices(parameter_array.shape)[0].flatten()
        p1 = np.indices(parameter_array.shape)[1].flatten()
        p2 = np.indices(parameter_array.shape)[2].flatten()
        col = parameter_array.flatten()

    plot_parameters_2D(p1, p2, col, ('p1_idx', 'p2_idx'), ax=fig.add_subplot(gs[0, 0]), lim_labels=lim_labels)
    plot_parameters_2D(p1, move, col, ('p1_idx', 'NumMoved'), ax=fig.add_subplot(gs[1, 0]), lim_labels=lim_labels)
    plot_parameters_2D(p2, move, col, ('p2_idx', 'NumMoved'), ax=fig.add_subplot(gs[2, 0]), lim_labels=lim_labels)

    clm = plot_parameters_3D(p1, p2, move, col, axis_labels=('p1_idx', 'p2_idx', 'NumMoved'),
                             ax=fig.add_subplot(gs[:, 1], projection="3d"), lim_labels=lim_labels)

    if not title:
        title = ""

    fig.suptitle(f"SREX Configuration: {title}", fontsize=16)
    fig.colorbar(clm, label="predicted probability")
    fig.tight_layout()
    fig.show()
