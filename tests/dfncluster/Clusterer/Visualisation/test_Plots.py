import os
import numpy as np
from dfncluster.Clusterer.Visualisation.Plots import plot_curve, plot_curve_compare


def test_plot_curve():
    target_file = 'single_dummy_plot.png'
    plot_curve(x=np.arange(100), filename_path=target_file)
    with open(target_file) as f:
        assert(not f.closed)
    assert(f.closed)
    os.remove(target_file)


def test_multiple_2d_plot():
    target_file = 'multiple_dummy_plot.png'
    domain = np.arange(start=-2*np.pi, stop=2*np.pi, step=np.pi/12)
    plot_curve_compare(
        domain, np.sin(domain), np.cos(domain), filename_path=target_file)
    with open(target_file) as f:
        assert(not f.closed)
    assert(f.closed)
    os.remove(target_file)
