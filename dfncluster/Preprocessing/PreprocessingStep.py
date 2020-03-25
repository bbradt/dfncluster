from abc import ABC, abstractmethod


class PreprocessingStep(ABC):

    def __init__(self):
        pass
    # abstract method

    def apply(self, x):
        return x


class PreprocessingSteps(PreprocessingStep):
    """
    Used for chaining several steps in a row
    """

    def __init__(self, *args):
        self.steps = args
        pass

    def apply(self, x):
        result = x
        for step in self.steps:
            print("Applying preprocessing step %s" % step.__class__.__name__)
            result = step.apply(result)
        return result
