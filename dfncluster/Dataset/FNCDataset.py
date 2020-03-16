import numpy as np
from dfncluster.Dataset import MatDataset


def corr_wrapper(x):
    return np.corrcoef(x)


class FNCDataset(MatDataset):
    """
        FNCDataset class. Extends the MatDataset class. See the docstring for Dataset for inherited fields/functions.
        Generates a dataset based on a CSV or TSV file, where features are given as .mat files.
        Adds additional functionality by generating windows.
        Each FNCDataset has:
            All inherited fields from FileFeatureDataset
            subject_data
            subject_labels
            subjects
            exemplars
        Each FNCDataset does:
            All inherited functions from FileFeatureDataset
    """

    def __init__(self, time_index=0, metric=corr_wrapper, window_size=22, **kwargs):
        """
            Constructor for FNCDataset.
            Usage:
                dataset = FNCDataset(time_index=0,
                                     metric=np.corrcoef,
                                     window_size=22,
                                     shuffle=True,
                                     feature_columns=['feature_1','feature_2'],
                                     label_columns=['label'],
                                     **kwargs)
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
        self.subject_data = None
        self.subject_labels = None
        self.subjects = None
        self.exemplars = None
        super(FNCDataset, self).__init__(metric=metric, time_index=time_index, window_size=window_size, **kwargs)

    def generate(self, **kwargs):
        """
            Generate FNCDataset with a sliding window over the temporal dimension,
            and computing a distance metric over the temporal dimension.
            Usage:
                dataset = FNCDataset(time_index=0,
                                     metric=np.corrcoef,
                                     window_size=22,
                                     shuffle=True,
                                     feature_columns=['feature_1','feature_2'],
                                     label_columns=['label'],
                                     **kwargs)
                dataset.generate()
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
        metric = kwargs['metric']
        window_size = kwargs['window_size']
        time_index = kwargs['time_index']
        x, y = super(FNCDataset, self).generate(**kwargs)
        # input(np.sum(np.isnan(x)))
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
                #input(str((np.sum(np.isnan(fnc)), fnc, window)))
                fnc = fnc[np.triu_indices_from(fnc)].flatten()
                variance_windows.append(np.var(fnc))
                sub_x.append(fnc)
                sub_y.append(y[i])
            _, indices = self.local_maxima(np.array(variance_windows))
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
        return matches

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
