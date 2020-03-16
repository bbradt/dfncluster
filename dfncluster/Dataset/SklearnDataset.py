import numpy as np
import inspect
from dfncluster.Dataset import Dataset
import sklearn.datasets as skd
from copy import deepcopy

# The following functions are either not used for loading data
# or only generate features. TODO: allow feature-only generation.
DISALLOWED_FUNCTIONS = [
    'clear_data_home',
    'dump_svmlight_file',
    'get_data_home',
    'make_biclusters',
    'make_checkerboard',
    'make_spd_matrix',
    'make_low_rank_matrix',
    'make_sparse_coded_signal',
    'make_sparse_spd_matrix',
    'make_spd_matrix'
]
SKLEARN_DATASETS = {n: getattr(skd, n) for n in dir(skd) if (n[0] != '_' and n not in DISALLOWED_FUNCTIONS)}
DEFAULT_DATASET = 'make_moons'


class SklearnDataset(Dataset):
    """
        SklearnDataset class. Wraps sklearn data loaders, and adds our 
            bundled in Dataset functionality.
        Each Dataset has:
            All fields from the Dataset superclass
        Each Dataset does:
            All functions from Dataset superclass
            ---
            function name   |   return type         |   args                |   kwargs              |   description
            ----------------|-----------------------|-----------------------|-----------------------|--------------------
            __init___       |   Dataset             |   self                |   shuffle, **kwargs   |   class constructor
            generate        |   (ndarray, ndarray)  |   self                |   **kwargs            |   generates the dataset (features, labels)
            resolve_kwargs  |   dict                | dataset_method,kwargs |   -                   |   sklearn only supports some kwargs, so we need to match those.
    """

    def __init__(self, dataset_name=DEFAULT_DATASET, dataset_method=None, **kwargs):
        """
            Constructor for SklearnDataset super-class. Generally should not be instantiated.
            Usage:
                dataset = SklearnDataset(dataset_name='iris',**kwargs)
            Kwargs:
                keyword         |   type        |   default     | Description                                    
                ----------------|---------------|---------------|-------------------
                dataset_name    |   str         |   'moons'     | the name of the dataset to loads, the name of the function to use
                dataset_method  |   function    |   None        | function handle for dataset loading directly.    
                NOTE: Other kwargs are passed to the self.generate function
            Args:
                -
            Return:
                Instantiated Dataset Object
        """
        if dataset_method is None:
            if dataset_name not in SKLEARN_DATASETS.keys():
                raise(Exception("The Dataset %s is not supported in sklearn" % dataset_name))
            dataset_method = SKLEARN_DATASETS.get(dataset_name)
        print(dataset_name)
        super_kwargs = self.resolve_kwargs(Dataset.__init__, kwargs)
        method_kwargs = deepcopy(kwargs)
        method_kwargs['return_X_y'] = True
        method_kwargs = self.resolve_kwargs(dataset_method, method_kwargs)
        super(SklearnDataset, self).__init__(dataset_method=dataset_method, **super_kwargs, **method_kwargs)

    def resolve_kwargs(self, dataset_method, kwargs):
        """
            Discard any kwargs which are incompatible with the sklearn function.
        """
        return {k: v for k, v in kwargs.items() if k in inspect.getfullargspec(dataset_method).args}

    def generate(self, **kwargs):
        """Wrap the sklearn dataset generation function.
        NOTE: All kwargs passed here are passed to that sklearn method.
        """
        dataset_method = kwargs.pop('dataset_method')
        X, y = dataset_method(**kwargs)
        return X, y
