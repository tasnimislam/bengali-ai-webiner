import pandas as pd
import numpy as np
from PIL import Image
import glob, os
from torch.utils.data import DataLoader, Dataset

def to_one_hot(value,size):
    np_one_hot = np.zeros(shape= size)
    np_one_hot[value] = 1
    return np_one_hot
class DatasetLoader(Dataset):
    def __init__(self, image_paths, labels_csv_path, size, convert='L'):
        self.image_paths = image_paths
        self.convert = convert
        self.size = size
        self.labels_csv_path = labels_csv_path
        self.df = pd.read_csv(labels_csv_path)

    def __getitem__(self, index):
        imageFile = Image.open(os.path.join('../dataTemp/IUB', self.image_paths[index])).convert(self.convert)
        imageFile = imageFile.resize((self.size, self.size), Image.ANTIALIAS) #Image.ANTIALIAS is the interpolaton method, you don't need to worry about that unless you are Jeff.
        label = to_one_hot(
                self.df.loc[self.df['fileID'] == self.image_paths[index]]['targetVar'].values,
                32
        )
        return imageFile, label
    def value(self, index):
        return DatasetLoader(self.image_paths, self.labels_csv_path, self.size).__getitem__(index)

    def __len__(self):
        return len(self.image_paths)
if __name__ == "__main__":
    imageFiles = os.listdir('../dataTemp/IUB')
    datasetTrain = DatasetLoader(imageFiles, '../dataTemp/groundTruth.csv', 224)
    test = datasetTrain.value(0)
