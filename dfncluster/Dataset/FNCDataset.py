"""
    Dataset for FNC data.
    Creates Correlation matrices for ICA timecourses
"""

import numpy as np
from dfncluster.Dataset import MatDataset

def corr_wrapper(x):
    return np.corrcoef(x)

class FNCDataset(MatDataset):
    def __init__(self, time_index=0, metric=corr_wrapper, window_size=22, **kwargs):
        super(FNCDataset, self).__init__(metric=metric, time_index=time_index, window_size=window_size, **kwargs)

    def generate(self, **kwargs):
        """
            Generate FNCDataset with a sliding window over the temporal dimension,
            and computing a distance metric over the temporal dimension.
        """
        metric = kwargs['metric']
        window_size = kwargs['window_size']
        time_index = kwargs['time_index']
        x, y = super(FNCDataset, self).generate(**kwargs)
        self.subject_data = x
        self.subject_labels = y
        exm_x = []
        exm_y = []
        new_x = []
        new_y = []
        subject_indices = []
        for i, xi in enumerate(x):
            sub_x = []
            sub_y = []
            variance_windows = []
            if time_index != 0:
                xi = xi.T
            for ti in range(xi.shape[0]-window_size):
                window = xi[ti:(ti+window_size)]
                fnc = metric(window.T)
                fnc = fnc[np.triu_indices_from(fnc)].flatten()
                variance_windows.append(np.var(fnc))
                sub_x.append(fnc)
                sub_y.append(y[i])
            maxima, indices = self.local_maxima(np.array(variance_windows))
            exm_x += [sub_x[j] for j in indices]
            exm_y += [sub_y[j] for j in indices]
            new_x += sub_x
            new_y += sub_y
            subject_indices += [i for x in sub_x]
        self.exemplars = dict(x=np.array(exm_x), y=np.array(exm_y))
        self.subjects = np.array(subject_indices)
        print('Found %d maxima' % (len(exm_x)))
        return np.array(new_x), np.array(new_y)

    def local_maxima(self, a, x=None, indices=True):
            """
            Finds local maxima in a discrete list 'a' to index an array 'x'
            if 'x' is not specified, 'a' is indexed.
            https://stackoverflow.com/questions/4624970/
                    finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
            """
            asm = self.smooth(a)
            maxima = [asm[i] for i in np.where(np.array(np.r_[1, asm[1:] < asm[:-1]] &
                                                        np.r_[asm[:-1] < asm[1:], 1]))[0]]
            matches = [self.find_nearest(a, maximum) for maximum in maxima]
            indices = [i for i in range(len(a)) if a[i] in matches]
            if indices:
                return matches, indices
            return matches

    def find_nearest(self, array, value):
            idx = (np.abs(array-value)).argmin()
            return array[idx]

    def get_subject_indices(self):
        return self.subjects[self.idx]
    
    def get_subjects(self):
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
        


