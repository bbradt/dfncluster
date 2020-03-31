import numpy as np
from dfncluster.Dataset import Dataset


class GaussianConnectivityDataset(Dataset):
    """
        GaussianDataset class. Extends Dataset superclass.
        Generates gaussian features with parameters per class.
        Each GaussianDataset has:
            All fields from Dataset
        Each GaussianDataset does:
            All methods from Dataset
    """

    def __init__(self,
                 num_ics=50,
                 num_subjects=300,
                 num_classes=2,
                 num_features=159,
                 sigma_ics=0.1,
                 sigma_ext=0.001,
                 epsilon=1):
        """
            Constructor for GaussianConnectivityDataset.
            Usage:
                dataset = GaussianConnectivityDataset(shuffle=True, num_ics=10, num_subjects=64, num_features=10, num_classes=2, **kwargs)
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                num_features|   int         |   2           |   the number of gaussian features to generate for all classes
                NOTE: Other kwargs are passed to the self.generate function
            Args:
                -
            Return:
                Instantiated GaussianDataset Object
        """
        super(GaussianConnectivityDataset, self).__init__(
            num_ics=num_ics,
            num_subjects=num_subjects,
            num_classes=num_classes,
            num_features=num_features,
            sigma_ics=sigma_ics,
            sigma_ext=sigma_ext,
            epsilon=epsilon,
        )

    def generate(self,  **kwargs):
        """
            Usage:
                dataset = GaussianConnectivityDataset(shuffle=True, num_ics=10, num_subjects=64, num_features=10, num_classes=2, **kwargs)
                dataset.generate()  # redundant since generate is called
            Kwargs:
                keyword     |   type        |   default     |       Description                                    
                num_features|   int         |   100           |   the number of gaussian features to generate for all classes
            Args:
                -
            Return:
                name    |   type    |   shape                       |   Description
                --------|-----------|-------------------------------|-----------------------
                x       |   ndarray |   instances x features x ...  |  the dataset features
                y       |   ndarray |   instances x labels x ...    |  the dataset labels
            End-State:
                -
            #TODO: Make this a class method?
        """
        num_ics = kwargs['num_ics']
        num_subjects = kwargs['num_subjects']
        num_classes = kwargs['num_classes']
        epsilon = kwargs['epsilon']
        sigma_ext = kwargs['sigma_ext']
        sigma_ics = kwargs['sigma_ics']
        num_features = kwargs['num_features']
        num_created = 0
        features = []
        labels = []
        connected_per_class = int(num_ics/num_classes)
        #class_signals = []
        # for c in range(num_classes):
        #    class_signals.append(np.random.normal(loc=0, scale=sigma_ics, size=(1, num_features)))
        while num_created < num_subjects:
            source_signal = np.random.normal(loc=0, scale=sigma_ics, size=(1, num_features))
            label = np.random.randint(0, num_classes)
            connected_start = label*connected_per_class
            connected_end = connected_start + connected_per_class
            signal = np.random.normal(loc=0, scale=sigma_ext, size=(num_ics, num_features))
            for ic in range(connected_start, connected_end):
                signal[ic, ...] = source_signal + np.random.normal(loc=0, scale=epsilon, size=source_signal.shape)
            labels.append(label)
            features.append(signal)
            num_created += 1
            print("Created subject %d with label %d" % (num_created, label))
        features = np.array(features)
        labels = np.array(labels)
        print(labels.shape, features.shape)
        return features, labels.flatten()
