import os
import pandas as pd
import torch
from torch.utils.data import dataset
from skimage import io

class cancer_images(dataset):
    def __init__(self, csv_file, rootdir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.rootdir = rootdir
        self.transform = transform

    def len(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.rootdir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        categor = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return image, categor