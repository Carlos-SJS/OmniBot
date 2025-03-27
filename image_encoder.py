import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()

        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        self.vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        feature_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Identity()
        
        for param in self.vit.parameters():
            param.requires_grad = False

        self.preprocessor = weights.transforms()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )

    def preprocess_image(self, img):
        return self.preprocessor(img).unsqueeze(0)

    def forward(self, x):
        x = x.to(torch.float32)

        x = self.vit(x)
        x = self.mlp(x)
        return x
