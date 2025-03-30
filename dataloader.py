import os
import torch
from PIL import Image as im
from torchvision import transforms as T
from torch.utils.data import Dataset

import pandas as pd

class OmniWorldLoader(Dataset):
    
    def __init__(self, image_directory, data_file, validation = False):
        self.loc_data = pd.read_csv(data_file)
        self.imgage_paths, self.coordinates = self.load_dataset(image_directory)

        if validation:
            self.create_preprocessor_val()
        else:
            self.create_preprocessor_train()
        
    def create_preprocessor_train(self):
        self.img_preprocessor = T.Compose([
            T.RandomResizedCrop(256),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.5),
            T.ToTensor(), 
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
    
    def create_preprocessor_val(self):
        self.img_preprocessor = T.Compose([
            T.Resize(256),
            T.ToTensor(), 
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def load_dataset(self, image_directory):
        coordinates = []
        img_paths = []
        for i, data in self.loc_data.iterrows():
            coordinates.append((data["lat"], data["lng"]))
            img_paths.append(os.path.join(image_directory, f"{i}.jpg"))
        
        return img_paths, coordinates

    def __len__(self):
        return self.loc_data.shape[0]

    def __getitem__(self, idx):
        loc = torch.tensor(self.coordinates[idx], dtype=torch.float)
        img = im.open(self.imgage_paths[idx]).convert('RGB').crop((20, 16, 620, 616))
        img = self.img_preprocessor(img)

        return img, loc