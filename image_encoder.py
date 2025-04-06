import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms as T

import timm
from transformers import CLIPModel, AutoProcessor
import open_clip

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=512):
        super(ImageEncoder, self).__init__()
        
        # self.backbone = self.backbone = timm.create_model(
        #     'swinv2_base_window16_256.ms_in1k',
        #     pretrained=True
        # )
        # self.backbone.reset_classifier(0)
        # feature_dim = self.backbone.num_features

        self.backbone, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='datacomp_l_s1b_b8k')
        feature_dim = 512

        # self.backbone = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # feature_dim = self.backbone.heads.head.in_features
        # self.backbone.heads = nn.Identity()
        
        #self.backbone = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        #feature_dim = 768

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            #nn.Linear(1024, 512),
            #nn.ReLU(),
            nn.Linear(1024, embedding_dim)
        )
    
    def to(self, device):
        self.backbone.to(device)
        self.mlp.to(device)
        return super().to(device)

    def forward(self, x):
        # x = self.backbone(x)
        # x = self.backbone.get_image_features(pixel_values=x)
        x = self.backbone.encode_image(x)
        x = self.mlp(x)

        return x