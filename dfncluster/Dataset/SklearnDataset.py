import numpy as np
import inspect
from dfncluster.Dataset import Dataset
import sklearn.datasets as skd

#  Wraps the SKLEARN module to make accessible via string
SKLEARN_DATASETS = {n: getattr(skd, n) for n in dir(skd) if '__' not in n}
DEFAULT_DATASET = 'make_moons'


class SklearnDataset(Dataset):
    def __init__(self, dataset_name=DEFAULT_DATASET, dataset_method=None, **kwargs):
        if dataset_method is None:
            if dataset_name not in SKLEARN_DATASETS.keys():
                raise(Exception("The Dataset %s is not supported in sklearn" % dataset_name))
            dataset_method = SKLEARN_DATASETS.get(dataset_name)
        kwargs['return_X_y'] = True
        kwargs = self.resolve_kwargs(dataset_method, kwargs)
        super(SklearnDataset, self).__init__(dataset_method=dataset_method, **kwargs)

    def resolve_kwargs(self, dataset_method, kwargs):
        """
            Discard any kwargs which are incompatible with the sklearn function.
        """
        return {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(dataset_method).args}

    def generate(self, **kwargs):
        dataset_method = kwargs.pop('dataset_method')
        X, y = dataset_method(**kwargs)
        return X, y
