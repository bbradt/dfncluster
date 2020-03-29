import numpy as np
import copy
from dfncluster.Preprocessing import PreprocessingStep
from multiprocessing import Pool


class NiftiPreprocessing(PreprocessingStep):
    def __init__(self):
        super(NiftiPreprocessing, self).__init__()

    def apply(self, x):
        """
            args:
                x - (N, X, Y, Z, T)
        """
        masks = []
        subjects = []
        for i, instance in enumerate(x):
            print("\tflattening, and computing mask for subject %s/%s" % (i, x.shape[0]))
            flattened = NiftiPreprocessing.flatten(instance)
            subjects.append(flattened)
            masks.append(NiftiPreprocessing.compute_subject_mask(flattened))
        print("\tcomputing global mask")
        global_mask = NiftiPreprocessing.compute_group_mask(masks)
        self.global_mask = global_mask
        self.local_masks = np.stack(masks, 0)
        for i, subject in enumerate(subjects):
            print("\tmasking and demeaning subject %s/%s" % (i, len(subjects)))
            masked = NiftiPreprocessing.apply_mask(subject, global_mask)
            subjects[i] = NiftiPreprocessing.remove_mean(masked)
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
        return subject_data - np.mean(subject_data, ax)[:, np.newaxis]

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
            sub_mask = np.array(subject_data[:, t] > voxel_mean[t], dtype=int)
            mask *= sub_mask.reshape(voxel, 1)
        return mask

    @staticmethod
    def compute_group_mask(masks, threshold=0.4):
        """
            args:
                masks
        """
        global_mask = np.mean(np.stack(masks, 0), 0)
        global_mask[global_mask < threshold] = 0
        global_mask[global_mask > threshold] = 1
        return global_mask

    @staticmethod
    def apply_mask(subject_data, mask):
        """
            args:
                masks
        """
        return subject_data[np.array(mask, dtype=bool).flatten(), :]
