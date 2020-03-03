import json
from dfncluster.Dataset import CsvDataset

if __name__=='__main__':
    dataset = CsvDataset(filename="dfncluster/data/CsvDatasets/IrisDataset/data.csv",
                         feature_columns=['sepal_length','sepal_width','petal_length','petal_width'],
                         label_columns=['species'])
    dataset.save('dfncluster/data/CsvDatasets/IrisDataset/iris_dataset')