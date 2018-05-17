from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import pandas as pd
import os


class Place365Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, loader=default_loader):
        self.cvs_file = pd.read_table(csv_file, sep=' ')
        self.image_dir = image_dir
        self.loader = loader
        self.transform = transform

    def __len__(self):
        n, _ = self.cvs_file.shape
        return n

    def __getitem__(self, index):
        row = self.cvs_file.iloc[index]
        image_path = os.path.join(self.image_dir, *row[0].split('/'))
        image = self.loader(image_path)
        target = row[1]
        if self.transform:
            image = self.transform(image)
        return image, target
