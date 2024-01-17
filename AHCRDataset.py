import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import cv2

class AHCRDataset(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        self.images_data = pd.read_csv(images_file, header=None).values
        self.labels_data = pd.read_csv(labels_file, header=None).values
        self.labels_data = self.labels_data - 1
        self.transform = transform

    def __len__(self):
        return len(self.images_data)

    def __getitem__(self, idx):
        image = self.images_data[idx].reshape(32, 32).astype(np.uint8)
        label = self.labels_data[idx]
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image.unsqueeze(1)).squeeze()
            image = image.numpy()
        return image, label

