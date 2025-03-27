import image_encoder
from geoclip import LocationEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToPILImage

import numpy as np
import pandas as pd
from PIL import Image as im


class OmniBot(nn.Module):
    def __init__(self):
        super().__init__()

        self.location_encoder = LocationEncoder()
        self.image_encoder = image_encoder.ImageEncoder()

        loc_data = pd.read_csv(r"C:\Users\carlo\Documents\Python\Geo\ds\dataset_generation\omniworld_locations.csv")
        coordinates = []
        for _, data in loc_data.iterrows():
            coordinates.append((data["lat"], data["lng"]))
        
        self.loc_galery = torch.tensor(coordinates, dtype=torch.float)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize to common input size
            transforms.ToTensor(),          # Convert PIL image to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])
    
    def to(self, device):
        self.image_encoder.to(device)
        self.location_encoder.to(device)
        return super().to(device)

    def forward(self, img, loc):
        img_embeddings = self.image_encoder(img)
        loc_embeddins = self.location_encoder(loc)

        img_embeddings = F.normalize(img_embeddings, dim=1)
        loc_embeddins = F.normalize(loc_embeddins, dim=1)

        return img_embeddings, loc_embeddins

    @torch.no_grad()
    def predict(self, img_path):
        loc_embeddings = self.location_encoder(self.loc_galery)
        loc_embeddings = F.normalize(loc_embeddings, dim=1)

        img = im.open(img_path).convert('RGB')
        img = self.transform(img).unsqueeze(0)

        img_embeddings = self.image_encoder(img)
        img_embeddings = F.normalize(img_embeddings, dim=1)

        logits = torch.matmul(img_embeddings, loc_embeddings.T)

        ix = np.argmax(logits)
        return (self.loc_galery[ix,0].item(), self.loc_galery[ix,1].item())

        