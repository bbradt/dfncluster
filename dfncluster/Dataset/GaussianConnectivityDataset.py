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
                 num_ics=10,
                 num_subjects=300,
                 num_features=159,
                 window_size=22,
                 class_parameters=[
                     dict(
                         states=[0.15, 0.15, 0.15, 0.4, 0.15],
                         sigma_ics=0.1,
                         sigma_ext=0.001,
                         epsilon=0.1,
                         transition_probability=[0.5, 0.5, 0.5, 0.25, 0.5]
                     ),
                     dict(
                         states=[0.4, 0.15, 0.15, 0.15, 0.15],
                         sigma_ics=0.1,
                         sigma_ext=0.001,
                         epsilon=0.1,
                         transition_probability=[0.25, 0.5, 0.5, 0.5, 0.5]
                     )
                 ],
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
        num_classes = len(class_parameters)
        super(GaussianConnectivityDataset, self).__init__(
            num_ics=num_ics,
            num_subjects=num_subjects,
            num_classes=num_classes,
            num_features=num_features,
            class_parameters=class_parameters,
            window_size=window_size
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
            # TODO: Make this a class method?
        """
        num_ics = kwargs['num_ics']
        num_subjects = kwargs['num_subjects']
        num_classes = kwargs['num_classes']
        class_parameters = kwargs['class_parameters']
        num_features = kwargs['num_features']
        window_size = kwargs['window_size']
        num_created = 0
        features = []
        labels = []
        # class_signals = []
        # for c in range(num_classes):
        #    class_signals.append(np.random.normal(loc=0, scale=sigma_ics, size=(1, num_features)))
        connected_states = []
        while num_created < num_subjects:
            label = np.random.randint(0, num_classes)
            parameters = class_parameters[label]
            sigma_ext = parameters['sigma_ext']
            sigma_ics = parameters['sigma_ics']
            epsilon = parameters['epsilon']
            states = parameters['states']
            transition_probability = parameters['transition_probability']
            num_states = len(states)
            state_vector = []
            while len(connected_states) < num_states:
                connected_states.append(np.random.choice(np.arange(num_ics), size=(np.random.randint(2, num_ics),)))

            source_signal = np.random.normal(loc=0, scale=sigma_ics, size=(1, num_features))
            signal = np.random.normal(loc=0, scale=sigma_ext, size=(num_ics, num_features))
            state_index = np.random.choice(np.arange(num_states), size=1, p=states)[0]
            actual_state = connected_states[state_index]
            for ww in range(0, num_features-window_size):
                source_window = source_signal[:, ww:(ww+window_size)]
                tprob = transition_probability[state_index]
                if np.random.randn() < tprob:
                    state_index = np.random.choice(np.arange(num_states), size=1, p=states)[0]
                    actual_state = connected_states[state_index]
                state_vector.append(state_index)
                for ic in range(num_ics):
                    if ic in actual_state:
                        signal[ic, ww:(ww+window_size)] = source_window+np.random.normal(loc=0, scale=epsilon, size=source_window.shape)
            labels.append(label)
            features.append(signal)
            num_created += 1
            print("Created subject %d with label %d" % (num_created, label))
        features = np.array(features)
        labels = np.array(labels)
        print(labels.shape, features.shape)
        return features, labels.flatten()
