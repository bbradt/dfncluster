import numpy as np
from sklearn.decomposition import PCA
from dfncluster.Preprocessing import PreprocessingStep
from dfncluster.Preprocessing.ICA.infomax.ica.ica import ica


class GroupICA:

    def apply(self, x):
        subjects = []
        for instance in x:
            subject_pcs = GroupICA.subject_level_pca(instance)
            subjects.append(subject_pcs.components_)
        stacked = GroupICA.stack_subjects(subjects)
        A, S = GroupICA.infomax(stacked, 100)
        return S

    @staticmethod
    def stack_subjects(subject_data):
        return np.hstack(subject_data)

    @staticmethod
    def subject_level_pca(subject_data, **kwargs):
        pca = PCA(**kwargs)
        return pca.fit(subject_data)

    @staticmethod
    def infomax(stacked_subjects, ncomp):
        model = ica(ncomp)
        A, S = model.fit(stacked_subjects)
        return A, S
