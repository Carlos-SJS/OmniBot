import os
import torch
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset

import pandas as pd

class OmniWorldLoader(Dataset):
    
    def __init__(self, image_directory, data_file):
        self.loc_data = pd.read_csv(data_file)
        self.imgage_paths, self.coordinates = self.load_dataset(image_directory)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to common input size
            transforms.ToTensor(),          # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])  # ImageNet normalization
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


        img = self.transform(img)

        return img, loc