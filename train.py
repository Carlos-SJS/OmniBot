import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_function(logits):
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    return F.cross_entropy(logits, labels)

def haversine_dist(lat1, lng1, lat2, lng2):
    R = 6371.0 #approximate earth radius (Km)

    lat1, lng1, lat2, lng2 = map(torch.deg2rad, (lat1, lng1, lat2, lng2))
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlng/2)**2
    return 2*R*torch.arcsin(torch.sqrt(a))

def distance_based_loss(logits, locs):
    lat, lng = locs[:,0:1], locs[:,1:2]
    lat1, lat2 = lat, lat.T
    lng1, lng2 = lng, lng.T

    dst_matrix = haversine_dist(lat1, lng1, lat2, lng2)
    softmax_logits = logits.softmax(dim=-1)

    return torch.mean(dst_matrix * softmax_logits)/1000


def train(train_dataloader, model, optimizer, epoch, device, temperature = 0.07):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    model.to(device)

    acc_loss = 0

    for i ,(imgs, locs) in bar:
        optimizer.zero_grad()
        
        imgs = imgs.to(device)
        locs = locs.to(device)

        #indices = torch.randint(0, model.loc_galery.size(0), (1000,))
        #rnd_locs = model.loc_galery[indices].to(device)
        #locs_all = torch.cat([locs, rnd_locs]).to(device)

        img_embeddings, loc_embeddings = model(imgs, locs)
        img_embeddings = F.normalize(img_embeddings, dim=1)
        loc_embeddings = F.normalize(loc_embeddings, dim=1)

        logits = img_embeddings @ loc_embeddings.T
        
        loss = distance_based_loss(logits, locs)

        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, acc_loss/(i+1)))