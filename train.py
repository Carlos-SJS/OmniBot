import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def info_nce_loss(image_embeds, loc_embeds, temperature=0.07):
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=1)
    loc_embeds = F.normalize(loc_embeds, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_embeds, loc_embeds.T) / temperature
    
    # Labels are the positives on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    # Compute loss from both directions (image→loc and loc→image)
    loss_i2l = F.cross_entropy(logits, labels)
    loss_l2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2l + loss_l2i) / 2

def train(train_dataloader, model, optimizer, epoch, device, temperature = 0.07):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    model.to(device)

    acc_loss = 0

    for i ,(imgs, locs) in bar:
        imgs = imgs.to(device)
        locs = locs.to(device)

        img_embeddings, loc_embeddings = model.forward(imgs, locs)
        loss = info_nce_loss(img_embeddings, loc_embeddings, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, acc_loss/(i+1)))