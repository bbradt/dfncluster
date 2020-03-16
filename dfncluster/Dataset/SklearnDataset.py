import numpy as np
import inspect
from dfncluster.Dataset import Dataset
import sklearn.datasets as skd

#  Wraps the SKLEARN module to make accessible via string
SKLEARN_DATASETS = {n: getattr(skd, n) for n in dir(skd) if '__' not in n}
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
        kwargs['return_X_y'] = True
        kwargs = self.resolve_kwargs(dataset_method, kwargs)
        super(SklearnDataset, self).__init__(dataset_method=dataset_method, **kwargs)

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
