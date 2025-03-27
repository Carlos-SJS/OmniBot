import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms as T

import timm

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        
        self.backbone = self.backbone = timm.create_model(
            'swinv2_base_window16_256.ms_in1k',
            pretrained=True
        )
        self.backbone.reset_classifier(0)

        feature_dim = self.backbone.num_features
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )

    def preprocess_image(self, img):
        return self.preprocessor(img).unsqueeze(0)

    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)

        return x
