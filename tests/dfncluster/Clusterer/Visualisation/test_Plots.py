import os
import numpy as np
import pandas as pd
from dfncluster.Clusterer.Metrics.TTest import t_test
from dfncluster.Clusterer.Visualisation.Plots import \
    plot_curve, plot_curve_compare, plot_multiple_ptest_results


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


def test_plot_multiple_ptest_results():
    # stage prepared assignments and label data representative of dFNC results
    assignments = np.loadtxt('assignments.csv')
    subject_labels = np.loadtxt('subject_labels.csv')

    # can change significance levels of test via p_level
    sig_levels = t_test(assignments, subject_labels, p_level=0.05)

    df = pd.DataFrame({

        # required argument to specify x-axis used across all subplots
        'time_window': np.arange(sig_levels.size),

        # after specifying time window, add as many as you want
        # going to use same data across all 'clustering algorithms'
        # for simplicity
        'kmeans': sig_levels,
        'gmm': sig_levels,
        'bayes': sig_levels,
        'dbscan': sig_levels,
        'foo': sig_levels,
        'bar': sig_levels,
        'dog': sig_levels,
        'oracle': sig_levels,
        'pokemon': sig_levels,
        'dragon': sig_levels,
        'bird': sig_levels,
        'rockets': sig_levels,
    })

    # plot the 12 subplots into a 3, 4 layout and specify figsize to increase size
    # and avoid tight subplot layouts
    # call must provide these arguments in order to allow for arbitrary amount of
    # clustering results
    plot_multiple_ptest_results(
        df, nrows=3, ncols=4,
        title='Example Plot Title', figsize=(16, 10),
        filename='test.png')

    with open('test.png') as f:
        assert(not f.closed)
    assert(f.closed)
    os.remove('test.png')
