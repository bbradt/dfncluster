import json
from dfncluster.Dataset import MatDataset


class UCLA:
    @staticmethod
    def make():
        dataset = MatDataset(filename="data/MatDatasets/UCLA/data.csv",
                             feature_columns=['task-rest_bold'],
                             label_columns=['diagnosis'])
        dataset.save('data/MatDatasets/UCLA/ucla')
        return dataset


if __name__ == '__main__':
    dataset = UCLA.make()
