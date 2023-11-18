from matplotlib.colors import Colormap
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

import torch


def plot_parameters_2D(
        x,
        y,
        col,
        axis_labels: Tuple[str, str],
        ax: Optional[plt.Axes] = None,
        cmap: Optional[Colormap] = None):
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not ax:
        _, ax = plt.subplots()

    ax.scatter(x, y, cmap=cmap, c=col, alpha=0.8, vmin=0, vmax=1)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])


def plot_parameters_3D(
        x,
        y,
        z,
        col,
        axis_labels: Tuple[str, str, str],
        ax: Optional[plt.Axes] = None,
        cmap: Optional[Colormap] = None):
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not ax:
        _, ax = plt.subplots()

    clm = ax.scatter(x, y, z, c=col, cmap=cmap, alpha=0.8, vmin=0, vmax=1)
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])

    return clm


def plot_srex_parameters(
        parameter_array,
        fig: Optional[plt.figure] = None,
        cmap: Optional[Colormap] = None,
        title: Optional[str] = None,
        only_pos: Optional[bool] = False,
) -> None:
    if not cmap:
        cmap = plt.colormaps["seismic_r"]

    if not fig:
        fig = plt.figure(figsize=(20, 12))

    gs = fig.add_gridspec(3, 2, width_ratios=(2 / 5, 3 / 5))

    move = np.indices(parameter_array.shape)[0]
    p1 = np.indices(parameter_array.shape)[1]
    p2 = np.indices(parameter_array.shape)[2]
    col = parameter_array.flatten()

    if only_pos:
        p1 = p1.clip(min=0.5)
        p2 = p2.clip(min=0.5)
        move = move.clip(min=0.5)

    plot_parameters_2D(p1, p2, col, ('p1_idx', 'p2_idx'), ax=fig.add_subplot(gs[0, 0]))
    plot_parameters_2D(p1, move, col, ('p1_idx', 'NumMoved'), ax=fig.add_subplot(gs[1, 0]))
    plot_parameters_2D(p2, move, col, ('p2_idx', 'NumMoved'), ax=fig.add_subplot(gs[2, 0]))

    clm = plot_parameters_3D(p1, p2, move, col, axis_labels=('p1_idx', 'p2_idx', 'NumMoved'),
                             ax=fig.add_subplot(gs[:, 1], projection="3d"))

    if not title:
        title = ""

    fig.suptitle(f"SREX Configuration: {title}", fontsize=16)
    fig.colorbar(clm, label="predicted probability")
    fig.tight_layout()
    fig.show()
