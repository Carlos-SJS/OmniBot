import image_encoder
from geoclip import LocationEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T

import numpy as np
import pandas as pd
from PIL import Image as im

class OmniBot(nn.Module):
    def __init__(self):
        super().__init__()

        self.location_encoder = LocationEncoder(from_pretrained=True)
        self.image_encoder = image_encoder.ImageEncoder()

        loc_data = pd.read_csv(r"C:\Users\carlo\Documents\Python\Geo\ds\dataset_generation\omniworld_locations.csv")
        coordinates = []
        for _, data in loc_data.iterrows():
            coordinates.append((data["lat"], data["lng"]))
        
        self.loc_gallery = torch.tensor(coordinates, dtype=torch.float)

        self.preprocessor_predict = T.Compose([
            T.Resize(224),
            T.ToTensor(), 
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        #for param in self.location_encoder.parameters():
        #    param.requires_grad = False
    
    def to(self, device):
        self.image_encoder = self.image_encoder.to(device)
        self.location_encoder = self.location_encoder.to(device)
        self.loc_gallery = self.loc_gallery.to(device)
        return super().to(device)


    def forward(self, img, loc):
        img_embeddings = self.image_encoder(img)
        loc_embeddins = self.location_encoder(loc)

        img_embeddings = F.normalize(img_embeddings, dim=1)
        loc_embeddins = F.normalize(loc_embeddins, dim=1)

        return img_embeddings, loc_embeddins

    @torch.no_grad()
    def predict(self, img_path):
        loc_embeddings = self.location_encoder(self.loc_gallery)
        loc_embeddings = F.normalize(loc_embeddings, dim=1)

        img = im.open(img_path).convert('RGB')
        img = self.preprocessor_predict(img).unsqueeze(0)

        img_embeddings = self.image_encoder(img)
        img_embeddings = F.normalize(img_embeddings, dim=1)

        logits = img_embeddings @ loc_embeddings.T

        ix = np.argmax(logits)
        return (self.loc_gallery[ix,0].item(), self.loc_gallery[ix,1].item())

        