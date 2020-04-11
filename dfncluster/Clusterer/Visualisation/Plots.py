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


def plot_multiple_ptest_results(df, nrows, ncols, title, figsize=None, filename=None):

    # Initialize Figure
    plt.style.use('seaborn-darkgrid')

    # Create a color-palette
    palette = ['b', 'g', 'r', 'c', 'm', 'k']

    if figsize is not None:
        plt.figure(figsize=figsize)

    subplot_num = 0
    for result in df.drop('time_window', axis=1):
        subplot_num += 1
        plt.subplot(nrows, ncols, subplot_num)

        # plot all results in grey-scale
        for ele in df.drop('time_window', axis=1):
            plt.plot(
                df['time_window'], df[ele],
                marker='', color='grey',
                linewidth=0.6, alpha=0.3)

        # plot specific algorithm results in bold color
        plt.plot(df['time_window'], df[result],
                 color=palette[subplot_num % len(palette)],
                 label=result,
                 linewidth=2.4, alpha=0.9)
        plt.title(result, loc='left',
                  color=palette[subplot_num % len(palette)],
                  fontsize=12, fontweight=0)

        if (subplot_num - 1) % ncols == 0:
            plt.yticks([0, 1])
            plt.ylabel('Significant')
        else:
            plt.tick_params(labelleft=False)

        if subplot_num <= (nrows - 1) * ncols:
            plt.tick_params(labelbottom=False)
        else:
            plt.xlabel('Time')

    plt.suptitle(title, fontsize=20)

    if filename is not None:
        plt.savefig(filename)
