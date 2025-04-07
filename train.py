import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

def loss_function(logits, _ = None):
    labels = torch.arange(logits.shape[0], device=logits.device)
    
    return F.cross_entropy(logits, labels)

def haversine_dist(lat1, lng1, lat2, lng2):
    R = 6371.0 #approximate earth radius (Km)

    lat1, lng1, lat2, lng2 = map(torch.deg2rad, (lat1, lng1, lat2, lng2))
    
    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = torch.sin(dlat/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin(dlng/2)**2
    return 2*R*torch.arcsin(torch.sqrt(a))

def distance_matrix(locs):
    lat, lng = locs[:,0:1], locs[:,1:2]
    lat1, lat2 = lat, lat.T
    lng1, lng2 = lng, lng.T

    return haversine_dist(lat1, lng1, lat2, lng2)
    dst_matrix = dst_matrix**2
    softmax_logits = logits.softmax(dim=-1)

    return torch.mean(softmax_logits @ dst_matrix)

def distance_based_loss(logits, locs, temperature = 75):
    dst_matrix = torch.exp(- distance_matrix(locs)/temperature)
    softmax_logits = logits.softmax(dim=-1)

    loss = -torch.mean(torch.log(softmax_logits + 1e-8) * dst_matrix[:softmax_logits.shape[0],:])
    return loss

def combined_loss(logits, locs):
    cross_entropy = loss_function(logits)
    dist_loss = distance_based_loss(logits, locs)/2
    return (cross_entropy + dist_loss)/2


def train(train_dataloader, model, optimizer, epoch, device, temperature = 0.07, loss_func = loss_function):
    print("Starting Epoch", epoch)

    bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
    model = model.to(device)

    acc_loss = 0

    for i ,(imgs, locs) in bar:
        optimizer.zero_grad()
        
        imgs = imgs.to(device)
        locs = locs.to(device)

        indices = torch.randint(0, model.loc_gallery.size(0), (4000,), device=device)
        rnd_locs = model.loc_gallery[indices]
        locs_all = torch.cat([locs, rnd_locs])

        img_embeddings, loc_embeddings = model.forward(imgs, locs_all)

        logits = (img_embeddings @ loc_embeddings.T)/temperature
        
        loss = loss_func(logits, locs_all)

        loss.backward()
        optimizer.step()

        acc_loss += loss.item()
        bar.set_description("Epoch {} loss: {:.5f}".format(epoch, acc_loss/(i+1)))