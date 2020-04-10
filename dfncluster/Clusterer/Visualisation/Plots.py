"""
   Plots:
        Simple plot utilities for generating dFNC clustering
        specific visualisations.
"""

import numpy as np
from matplotlib import pyplot as plt


def plot_curve(x, y=None, title='Y over X', x_label='x', y_label='y',
               curve_type='-', color='b', lw=2, grid_enable=True,
               filename_path=None):
    if y is None:
        plt.plot(np.arange(x.size), x, curve_type, color, lw)
    else:
        plt.plot(x, y, curve_type, color, lw)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid_enable)
    if filename_path is not None:
        plt.savefig(filename_path)


def plot_curve_compare(x, y1, y2, title='Ys over X',
                       x_label='x', y_label='y', grid_enable=True,
                       filename_path=None):
    plt.plot(x, y1, x, y2)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid_enable)
    if filename_path is not None:
        plt.savefig(filename_path)
