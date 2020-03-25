import numpy as np
from sklearn.decomposition import PCA
from dfncluster.Preprocessing import PreprocessingStep
from dfncluster.Preprocessing.ICA.infomax.ica.ica import ica
from multiprocessing import Pool


class GroupICA(PreprocessingStep):

    def __init__(self, workers=32):
        self.workers = 32
        super(GroupICA, self).__init__()

    def apply(self, x):
        self.subjects = []
        for instance in x:
            subject_pcs = GroupICA.subject_level_pca(instance)
            self.subjects.append(subject_pcs.components_.T)
        stacked, indices = GroupICA.stack_subjects(self.subjects)
        self.A, self.S = GroupICA.infomax(stacked.T, 100)
        return GroupICA.unstack_sources(self.S, indices)

    @staticmethod
    def stack_subjects(subject_data):
        return np.hstack(subject_data), np.cumsum([s.shape[1] for s in subject_data])

    @staticmethod
    def unstack_sources(sources, indices):
        i0 = 0
        subjects = []
        for i1 in indices:
            subjects.append(sources[:, i0:i1])
            i0 = i1
        return subjects

    @staticmethod
    def subject_level_pca(subject_data, **kwargs):
        pca = PCA(n_components=120)
        return pca.fit(subject_data.T)

    @staticmethod
    def infomax(stacked_subjects, ncomp):
        model = ica(ncomp)
        fitted = model.fit(stacked_subjects)
        return fitted.mix, fitted.sources
