import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import sys
import os
from scipy.stats import ttest_ind
from itertools import combinations
from sklearn.linear_model import LinearRegression


def corr_wrapper(x):
    return np.corrcoef(x)


class dFNC:
    def __init__(self, dataset=None, first_stage_algorithm=None, second_stage_algorithm=None, time_index=0, metric=corr_wrapper, window_size=22, **kwargs):
        """kwargs:
            dataset     FNCDataset  Some FNC Dataset
            clusterer   Clusterer   KMeansClusterer
        """
        self.dataset = dataset
        self.subject_data = self.dataset.features
        # input(str(self.subject_data.shape))
        self.subject_labels = self.dataset.labels
        self.first_stage_algorithm = first_stage_algorithm
        self.second_stage_algorithm = second_stage_algorithm
        self.results = []
        self.subjects = None
        self.exemplars = None
        self.metric = metric
        self.time_index = time_index
        self.window_size = window_size
        self.bad_indices = []

    def compute_windows(self, **kwargs):
        """
            Generate FNCDataset with a sliding window over the temporal dimension,
            and computing a distance metric over the temporal dimension.
            Usage:
                dfnc.compute_windows()
            Kwargs:
                keyword         |   type        |   default     |       Description
                ----------------|---------------|---------------|-------------------
                time_index      |   int         |   0           | shape index corresponding to temporal dimension
                metric          |   function    |   np.corrcef  | function for distance metric between two components (must return a matrix)
                window_size     |   int         |   22          | window size for sliding window analysis in dFNC
                NOTE: Other kwargs are passed to the CsvDataset superclass constructor (and thus to generate, and shuffle methods)
            Args:
                -
            Return:
                Instantiated FNCDataset Object
        """
        metric = self.metric
        window_size = self.window_size
        time_index = self.time_index
        exm_x = []
        exm_y = []
        new_x = []
        new_y = []
        subject_indices = []
        self.bad_indices = []
        mintime = np.min([x.shape[self.time_index] for x in self.subject_data])
        for i, xi in enumerate(self.subject_data):
            sub_x = []
            sub_y = []
            variance_windows = []
            if time_index != 0:
                xi = xi.T
            found_nan = np.isnan(xi).any() or np.isinf(xi).any()
            timedim = xi.shape[0]
            for ti in range(timedim-window_size):
                window = xi[ti:(ti+window_size)]
                fnc = metric(window.T)
                # input(fnc.shape)
                fnc = fnc[np.triu_indices_from(fnc)].flatten()
                found_nan = found_nan or np.isnan(fnc).any() or np.isinf(fnc).any()
                if found_nan:
                    sub_x = []
                    sub_y = []
                    break
                variance_windows.append(np.var(fnc))
                sub_x.append(fnc)
                sub_y.append(self.subject_labels[i])
            if found_nan:
                print('Found Nan or Inf in subject %d' % i)
                self.bad_indices.append(i)
                continue
            _, indices = self.local_maxima(np.array(variance_windows))
            exm_x += [sub_x[j] for j in indices]
            exm_y += [sub_y[j] for j in indices]
            new_x += sub_x
            new_y += sub_y
            subject_indices += [i for x in sub_x]
        self.exemplars = dict(x=np.array(exm_x), y=np.array(exm_y))
        self.subjects = np.array(subject_indices)
        return np.array(new_x), np.array(new_y)

    def local_maxima(self, a, x=None, indices=True):
        """
        Finds local maxima in a discrete list 'a' to index an array 'x'
        if 'x' is not specified, 'a' is indexed.
        https://stackoverflow.com/questions/4624970/
                finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
        Usage:
            C = np.array([1,2,3,4,5,4,3,2,1])
            maxima = local_maxima(C)
        Kwargs:
            keyword         |   type        |   default     |       Description                                    
            ----------------|---------------|---------------|-------------------
            x               | ndarray       |   None        |   not actually used
            indices         | boolean       |   True        |   return indices of matches
        Args:
            name    |   type    |  Description
            --------|-----------|-------------------------------
            a       | ndarray   |  array in which to detect local maxima
        Return:
            name    |   type    |   shape                       |   Description
            --------|-----------|-------------------------------|-----------------------
            matches | list<num> |  num_matches                  | a list of the matched maxima
            indices | list<int> |  num_matches                  | a list of the matched indices
        """
        asm = self.smooth(a)
        maxima = [asm[i] for i in np.where(np.array(np.r_[1, asm[1:] < asm[:-1]] &
                                                    np.r_[asm[:-1] < asm[1:], 1]))[0]]
        matches = [self.find_nearest(a, maximum) for maximum in maxima]
        indices = [i for i in range(len(a)) if a[i] in matches]
        if indices:
            return matches, indices
        return matches, indices

    def find_nearest(self, array, value):
        """
        find the first nearest value in an array according to the L1 norm
        Usage:
            C = np.array([1,2,3,4,5,4,3,2,1])
            x = 5.4
            nearest = dataset.find_nearest(C,x)
        Kwargs:
            -
        Args:
            name    |   type    |  Description
            --------|-----------|-------------------------------
            array   | ndarray   | input array
            value   | float     | value to find closest distance to
        Return:
            the matched value in the array
        TODO:
            Convert to staticmethod
        """
        idx = (np.abs(array-value)).argmin()
        return array[idx]

    def get_subject_indices(self):
        """
        Get the indices of FNC indices corresponding to particular subjects
        Usage:
            dataset.get_subject_indices()
        Kwargs:
            -
        Args:
            -
        Return:
            return the subject array indexed by idx
        """
        return self.subjects[self.idx]

    def get_subjects(self):
        """
        Get subjects and subject labels
        Usage:
            dataset.get_subjects()
        Kwargs:
            -
        Args:
            -
        Return:
            getter for subject_data and labels
        """
        return self.subject_data, self.subject_labels

    def smooth(self, x, window_len=4):
        """
        Smooths the window using np hanning

        args:
                x - the input signal
                window_len - length of the window
        """
        if x.ndim != 1:
            raise(ValueError("smooth only accepts 1 dimension arrays."))

        if x.size < window_len:
            raise(ValueError("Input vector needs to be bigger than window size."))

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        w = np.hanning(window_len)
        y = np.convolve(w/w.sum(), s, mode='valid')
        return y

    def eval_k_clusters(self, k, filename, **kwargs):
        if self.first_stage_algorithm is not None:
            fnc_features, fnc_labels = self.compute_windows()
            exemplar_clusterer = self.first_stage_algorithm(X=self.exemplars['x'], Y=self.exemplars['y'], **kwargs)
            exemplar_clusterer.evaluate_k(k, filename)

    def visualize_clusters(self, fnc_features, assignments, clusterer_name, filename, centroids=None):

        fnc_features_centered = fnc_features - np.mean(fnc_features, axis=0)

        U, S, Vt = np.linalg.svd(fnc_features_centered, full_matrices=False)

        S = np.diag(S)

        dim_reduced_X = fnc_features.dot(Vt.T)

        dim_reduced_X = dim_reduced_X[:, 0:3]  # take first 3 dimensions

        plt.figure(2, figsize=(8, 6))
        plt.clf()
        COLOR_LABELS = np.reshape(assignments, (-1))

        plt.scatter(dim_reduced_X[:, 0], dim_reduced_X[:, 1], c=COLOR_LABELS, marker='o', cmap='viridis')
        plt.legend(np.unique(COLOR_LABELS).tolist())

        if centroids is not None:

            centroids_centered = centroids - np.mean(centroids, axis=0)

            centroids_reduced_X = centroids.dot(Vt.T)

            centroids_reduced_X = centroids_reduced_X[:, 0:2]  # take first 2 dimensions

            plt.scatter(centroids_reduced_X[:, 0], centroids_reduced_X[:, 1], c='r', s=10**3, linewidth=3, marker='x')

        plt.xlabel('PCA Dim 1')
        plt.ylabel('PCA Dim 2')
        plt.title(clusterer_name)
        print('Created cluster visualization on full dataset.')
        sb.set()
        plt.savefig(filename, bbox_inches="tight")

        plt.close()
        plt.figure()
        plt.clf()
        fig = plt.figure(3, figsize=(8, 6))
        ax = fig.gca(projection='3d')
        scatter = ax.scatter(xs=dim_reduced_X[:, 0], ys=dim_reduced_X[:, 1], zs=dim_reduced_X[:, 2],
                             c=COLOR_LABELS, marker='o', cmap='rainbow', s=10, alpha=0.5)
        legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.tight_layout()
        ax.set_xlabel('PCA Dim 1')
        ax.set_ylabel('PCA Dim 2')
        ax.set_zlabel('PCA Dim 3')
        plt.title(clusterer_name)
        file_name, extension = os.path.splitext(filename)
        filename = file_name + "_3d" + extension
        plt.savefig(filename, bbox_inches="tight")
        print('Created 3d cluster visualization on full dataset.')

    def run(self, evaluate=False, grid_params={}, vis_filename="results/cluster_vis.png", state_filename="results/states_vis.png", ttest_fileprefix="results/ttest_", networks=None, **kwargs):
        """Run dFNC, including the following steps:
            1. Window computation
            2. First stage exemplar clustering
            3. Second stage full data clustering with exemplar initialization
            4. reassignment of cluster assignments to subjects
        """
        print("Computing FNC Windows")
        fnc_features, fnc_labels = self.compute_windows()

        if self.first_stage_algorithm is not None:
            print("Performing exemplar clustering")
            exemplar_clusterer = self.first_stage_algorithm(X=self.exemplars['x'], Y=self.exemplars['y'], param_grid=grid_params, **kwargs)
            exemplar_clusterer.model = exemplar_clusterer.fit_grid()
            exemplar_clusterer.fit()
            self.exemplar_clusterer = exemplar_clusterer
            if self.second_stage_algorithm is not None:
                print("Performing full clustering")
                kwargs['n_init'] = 1  # reset since we used exemplar to produce initial centers
                cluster_instance = self.second_stage_algorithm(
                    X=fnc_features, Y=fnc_labels,
                    initialization=exemplar_clusterer.get_results_for_init(), **kwargs)
                cluster_instance.fit()
                self.last_clusterer = cluster_instance
            else:
                cluster_instance = self.exemplar_clusterer
                self.last_clusterer = self.exemplar_clusterer
        elif self.second_stage_algorithm is not None:
            cluster_instance = self.second_stage_algorithm(
                X=fnc_features, Y=fnc_labels,
                **kwargs)
            cluster_instance.fit()
            self.last_clusterer = cluster_instance
        else:
            evaluate = False
            cluster_instance = None

        if evaluate:
            print("Evaluating clustering")
            cluster_instance.evaluate()

        if cluster_instance is not None:
            print("Reassigning states to subjects")
            assignments = self.reassign_to_subjects(
                cluster_instance.assignments, self.subjects)
            subject_windows = self.reassign_to_subjects(
                fnc_features, self.subjects
            )

            self.visualize_clusters(fnc_features, cluster_instance.assignments, kwargs['name'], vis_filename, cluster_instance.centroids)
            print("Collecting state statistics")
            class_centroids, beta_features, class_partitions, nc = self.collect_states(assignments, classes=self.dataset.labels,
                                                                                       subject_data=subject_windows, time_index=self.time_index)
            print("Visualizing States")
            self.visualize_states(assignments, class_centroids, class_partitions, nc, filename=state_filename,
                                  ttest_fileprefix=ttest_fileprefix, networks=networks)
            return cluster_instance.results, assignments, beta_features
        else:
            return None, fnc_features, None

    def reassign_to_subjects(self, cluster_assigments, subjects):
        reassigned = []
        for i in range(max(subjects)+1):
            reassignments = cluster_assigments[subjects == i]
            if len(reassignments) > 0:
                reassigned.append(reassignments)
        return np.array(reassigned)

    def line_search(self, line_params, **kwargs):
        """ Vary a particular parameter, and get clustering results when checking that parameter.
        """
        results = dict()
        assignments = dict()
        for param_name in line_params.keys():
            results[param_name] = dict()
            assignments[param_name] = dict()
            for param_val in line_params[param_name]:
                kwargs[param_name] = param_val
                results[param_name][param_val], assignments[param_name][param_val] = self.run(**kwargs)
        return results, assignments

    def collect_states(self, assignments, filename="results/states.png", classes=None, subject_data=None, networks=None, time_index=1):
        if time_index == 0:
            time_index = 2
        nc = self.subject_data.shape[time_index]
        try:
            states = np.unique(np.array(assignments).flatten())
        except ValueError:
            states = np.unique(np.concatenate([np.unique(a) for a in assignments]))
        num_states = len(states)
        class_labels = np.unique(classes)
        class_combos = list(combinations(class_labels, 2))
        num_classes = len(class_labels)
        class_centroids = dict()
        class_partitions = dict()
        beta_features = np.zeros((subject_data.shape[0], num_classes*num_states))

        for j, L in enumerate(class_labels):
            relevant_indices = [i for i in range(len(assignments)) if classes[i] == L]
            relevant_windows = np.stack(subject_data[relevant_indices], 0)
            relevant_windows = relevant_windows.reshape(relevant_windows.shape[0]*relevant_windows.shape[1], relevant_windows.shape[2])
            relevant_assignments = np.concatenate(assignments[relevant_indices]).flatten()
            class_centroids[L] = dict()
            class_partitions[L] = dict()
            for k in states:
                matched_windows = [relevant_windows[i, :] for i in range(len(relevant_assignments)) if relevant_assignments[i] == k]
                centroid_k = np.mean(matched_windows, 0)
                Z = np.zeros((nc, nc))
                Z[np.triu_indices(nc)] = centroid_k
                Z = Z.T
                Z[np.triu_indices(nc)] = centroid_k
                class_centroids[L][k] = centroid_k
                class_partitions[L][k] = matched_windows
        for i, xi in enumerate(subject_data):
            betas = np.zeros((xi.shape[0], num_classes, num_states))
            for j, L in enumerate(class_labels):
                for k in states:
                    centroid = class_centroids[L][k]
                    model = LinearRegression()
                    try:
                        model.fit(np.nan_to_num(xi.T), np.nan_to_num(centroid))
                        betas[:, j, k] = model.coef_
                    except TypeError:
                        continue
            betas = np.mean(betas, 0)
            beta_features[i, :] = betas.reshape(num_classes*num_states)
        return class_centroids, beta_features, class_partitions, nc

    def visualize_states(self, assignments, class_centroids, class_partitions, nc, filename="results/states.png", ttest_fileprefix="results/ttests_", class_names=None, networks=None, time_index=1):
        try:
            states = np.unique(np.array(assignments).flatten())
        except ValueError:
            states = np.unique(np.concatenate([np.unique(a) for a in assignments]))
        num_states = len(states)
        class_labels = list(class_centroids.keys())
        class_combos = combinations(class_labels, 2)
        num_classes = len(class_labels)
        sb.set()
        fig1, ax = plt.subplots(num_classes, num_states, figsize=(30, 10))
        state_max = -float("inf")
        state_min = float("inf")
        for L in class_labels:
            for k in states:
                centroid_k = class_centroids[L][k]
                mmax = np.max(centroid_k)
                mmin = np.min(centroid_k)
                if mmax > state_max:
                    state_max = mmax
                if mmin < state_min:
                    state_min = mmin
        for ck, L in enumerate(class_labels):
            for k in states:
                centroid_k = class_centroids[L][k]
                Z = np.zeros((nc, nc))
                Z[np.triu_indices(nc)] = centroid_k
                Z = Z.T
                Z[np.triu_indices(nc)] = centroid_k
                ax[ck, k].imshow(Z, cmap='jet', vmin=state_min, vmax=state_max)
                ax[ck, k].set_title("State %d" % k)
                ax[ck, k].set_xticks(())
                ax[ck, k].set_yticks(())
        plt.savefig(filename, bbox_inches='tight')
        vmin = float("inf")
        vmax = -float("inf")
        scores = dict()
        for ck, (k1, k2) in enumerate(class_combos):
            scores[ck] = dict()
            for k in states:
                states_1 = class_partitions[k1][k]
                states_2 = class_partitions[k2][k]
                ttest = ttest_ind(states_1, states_2, axis=0)
                score = np.log10(ttest.pvalue)*np.sign(ttest.statistic)
                scores[ck][k] = score
                mmin = np.min(score)
                mmax = np.max(score)
                if mmin < vmin:
                    vmin = mmin
                if mmax > vmax:
                    vmax = mmax
        for ck, (k1, k2) in enumerate(class_combos):
            fig2, ax = plt.subplots(1, len(states))
            ttest_filename = ttest_fileprefix + "%s-%s.png" % (k1, k2)
            for k in states:
                score = scores[ck][k]
                Z = np.zeros((nc, nc))
                Z[np.triu_indices(nc)] = score
                Z = Z.T
                Z[np.triu_indices(nc)] = score
                ax[k].imshow(Z, cmap='jet', vmin=vmin, vmax=vmax)
                ax[k].set_title("State %d/ TTest %s - %s" % (k, k1, k2))
                ax[k].set_xticks(())
                ax[k].set_yticks(())
            plt.savefig(ttest_filename, bbox_inches='tight')
        return fig1

    def save(self, filename):
        package = dict()
        package['first_stage_clusterer'] = self.exemplar_clusterer
        package['second_stage_clusterer'] = self.last_clusterer
        package['exemplars'] = self.exemplars
        package['subjects'] = self.subjects
        np.save(filename, package, allow_pickle=True)
