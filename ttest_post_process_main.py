import os
import re
import subprocess
import numpy as np
import pandas as pd
from dfncluster.Dataset import MatDataset
from dfncluster.Clusterer.Metrics.TTest import t_test
from dfncluster.Clusterer.Visualisation.Plots import \
    plot_multiple_ptest_results


def load_data():
    results = dict({cluster: np.load(file_path) for cluster, file_path in [
        (re.search('(?<=results/).*_ucla', filename).group(0).strip('_ucla'),
         filename) for filename in subprocess.run(
             ['find', '-name', '*_features*'],
             stdout=subprocess.PIPE).stdout.decode('utf-8').split('\n')
        [:-1]]})

    ucla_file = os.path.join('data', 'MatDatasets', 'UCLA', 'ucla.npy')
    labels = MatDataset.load(ucla_file).labels

    return results, labels


def main():

    results, labels = load_data()
    del results['dbscan']

    df = pd.DataFrame({
        cluster: t_test(data['assignments'], labels, p_level=0.10)
        for cluster, data in results.items()
    })
    plot_multiple_ptest_results(
        df, nrows=2, ncols=2,
        title='UCLA Assignment T-Test Comparisons', figsize=(16, 10),
        feature_domain='Time',
        filename='assignment_t_test_visualization.png')

    df = pd.DataFrame({
        cluster: t_test(data['betas'], labels, p_level=0.10)
        for cluster, data in results.items()
    })
    plot_multiple_ptest_results(
        df, nrows=2, ncols=2,
        title='UCLA Betas T-Test Comparisons', figsize=(16, 10),
        feature_domain='Beta Coefficient',
        filename='beta_t_test_visualization.png')


if __name__ == "__main__":
    main()
