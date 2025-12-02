import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class CassavaDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with image_id and label.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        
        # Use PIL for compatibility with torchvision transforms
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # Return a black image or handle error appropriately
            image = Image.new('RGB', (224, 224))

        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

class OODDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with OOD images.
            transform (callable, optional): Optional transform.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        
        try:
            image = Image.open(img_name).convert('RGB')
        except:
             image = Image.new('RGB', (224, 224))
             
        if self.transform:
            image = self.transform(image)

        # Label -1 for OOD
        return image, -1
