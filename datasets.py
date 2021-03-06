from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class DomainDataset(Dataset):
    def __init__(self, image_dir, annotations_file_path, transform = None):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(annotations_file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")
        y_label = torch.tensor(float(self.annotations.iloc[index, 1]))
        if self.transform is not None:
            img = self.transform(img)
            
        return img, y_label