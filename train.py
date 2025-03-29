import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_function(image_embeds, loc_embeds):
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=1)
    loc_embeds = F.normalize(loc_embeds, dim=1)
    
    # Compute similarity matrix
    
    #logits = torch.matmul(image_embeds, loc_embeds.T)
    logits = F.cosine_similarity(image_embeds.unsqueeze(1), loc_embeds.unsqueeze(0), dim=-1)
    # Labels are the positives on the diagonal
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    return F.cross_entropy(logits, labels)

def train(train_dataloader, model, optimizer, epoch, device, temperature = 0.07):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    model.to(device)

    acc_loss = 0

    for i ,(imgs, locs) in bar:
        imgs = imgs.to(device)
        locs = locs.to(device)

        #indices = torch.randint(0, model.loc_galery.size(0), (1000,))
        #rnd_locs = model.loc_galery[indices].to(device)
        #locs_all = torch.cat([locs, rnd_locs]).to(device)

        img_embeddings, loc_embeddings = model(imgs, locs)
        loss = loss_function(img_embeddings, loc_embeddings)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, acc_loss/(i+1)))