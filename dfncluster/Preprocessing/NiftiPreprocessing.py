import numpy as np

from dfncluster.Preprocessing import PreprocessingStep


class NiftiPreprocessing(PreprocessingStep):
    def __init__(self):
        pass

    def apply(self, x):
        """
            args:
                x - (N, X, Y, Z, T)
        """
        masks = []
        subjects = []
        for instance in x:
            flattened = NiftiPreprocessing.flatten(instance)
            demeaned = NiftiPreprocessing.remove_mean(flattened)
            subjects.append(demeaned)
            masks.append(NiftiPreprocessing.compute_subject_mask(demeaned))
        global_mask = NiftiPreprocessing.compute_group_mask(masks)
        for i, subject in enumerate(subjects):
            subjects[i] = NiftiPreprocessing.apply_mask(subject, global_mask)
        return np.stack(subjects, 0)

    @staticmethod
    def flatten(subject_data):
        """
            args:
                subject_data - (X, Y, Z, T) 4-d array
                             - (X*Y*Z, T) 2-d array
        """
        if subject_data.ndim == 4:
            x, y, z, t = subject_data.shape
            return subject_data.reshape(x*y*z, t)
        elif subject_data.ndim == 3:
            x, y, z = subject_data.shape
            return subject_data.reshape(x*y*z, 1)
        else:
            return subject_data

    @staticmethod
    def remove_mean(subject_data, ax=1):
        """
            args:
                subject_data - (X*Y*Z, T) 2-d array
        """
        return subject_data - np.mean(subject_data, ax)

    @staticmethod
    def compute_subject_mask(subject_data):
        """
            args:
                subject_data - (X*Y*Z, T) 2-d array
        """
        voxel, time = subject_data.shape
        voxel_mean = np.mean(subject_data, 0)
        mask = np.ones((voxel, 1))
        for t in range(time):
            mask *= subject_data[:, t] > voxel_mean
        return mask

    @staticmethod
    def compute_group_mask(masks):
        """
            args:
                masks
        """
        global_mask = np.ones_like(masks[0].shape)
        for mask in masks:
            global_mask *= mask
        return global_mask

    @staticmethod
    def apply_mask(subject_data, mask):
        """
            args:
                masks
        """
        return subject_data[:, mask]
