import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os
import random

class LeafDataset(Dataset):
    
    def __init__(self, csv_file, imgs_path, transform=None):
        self.df = pd.read_csv(csv_file) 
        self.imgs_path = imgs_path 
        self.transform = transform 
        self.len = self.df.shape[0] 
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index): 
        row = self.df.iloc[index]
        image_path = self.imgs_path + row[0]
        image = torchvision.io.read_image(image_path).float()
        target = torch.tensor(row[-6:], dtype=torch.float)
        if self.transform:
            return self.transform(image), target
        return image, target

class TripletLeafDataset(Dataset):
    def __init__(self, csv_file, imgs_path, transform=None):
        self.df = pd.read_csv(csv_file)
        self.imgs_path = imgs_path
        self.transform = transform
        self.len = self.df.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        anchor_row = self.df.iloc[index]
        anchor_image_path = os.path.join(self.imgs_path, anchor_row[0])
        anchor_image = torchvision.io.read_image(anchor_image_path).float()
        anchor_target = torch.tensor(anchor_row[-6:], dtype=torch.float)

        # Choose a positive example (same class as anchor)
        positive_candidates = self.df[self.df.iloc[:, -6:].eq(anchor_row[-6:]).all(axis=1)]
        positive_row = positive_candidates.iloc[random.randint(0, len(positive_candidates) - 1)]
        positive_image_path = os.path.join(self.imgs_path, positive_row[0])
        positive_image = torchvision.io.read_image(positive_image_path).float()
        positive_target = torch.tensor(positive_row[-6:], dtype=torch.float)

        # Choose a negative example (different class than anchor)
        negative_candidates = self.df[~self.df.iloc[:, -6:].eq(anchor_row[-6:]).all(axis=1)]
        negative_row = negative_candidates.iloc[random.randint(0, len(negative_candidates) - 1)]
        negative_image_path = os.path.join(self.imgs_path, negative_row[0])
        negative_image = torchvision.io.read_image(negative_image_path).float()
        negative_target = torch.tensor(negative_row[-6:], dtype=torch.float)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return {"anchor": anchor_image, "positive": positive_image, "negative": negative_image}