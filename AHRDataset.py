from  torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class AHRDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images_data = pd.read_csv(images_file, header=None).values
        self.labels_data = pd.read_csv(labels_file, header=None).values - 1
        self.transform = transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        image = self.images_data[idx].reshape(32, 32).astype(np.uint8)
        label = self.labels_data[idx]

        
        if self.transform:
            image = self.transform(image)

        return image, label