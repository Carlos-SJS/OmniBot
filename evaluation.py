import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from geopy import distance

def eval_model(model, val_dataloader, device='cpu'):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        locs = model.loc_gallery.to(device)
        loc_embeds = model.location_encoder(locs)
        loc_embeds = F.normalize(loc_embeds, dim = 1)

        for imgs, labels in tqdm(val_dataloader, desc="Evaluating"):
            labels = labels.cpu().numpy()
            imgs = imgs.to(device)

            img_embeds = model.image_encoder(imgs)
            img_embeds = F.normalize(img_embeds, dim = 1)

            logits = img_embeds @ loc_embeds.T
            outs = torch.argmax(logits, dim=-1).detach().cpu().numpy()

            preds.append(outs)
            targets.append(labels)

    model.train()

    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    distance_thresholds = [2500, 750, 200, 25, 1]
    dst_counts = [0 for _ in range(len(distance_thresholds)+1)]

    avg_dist = 0

    locs = model.loc_gallery.to('cpu').numpy()

    distance_thresholds = [100000, 2500, 750, 200, 25, 1]
    dst_counts = [0 for _ in range(len(distance_thresholds))]

    avg_dist = 0

    for t, p in zip(targets, preds):
        dist = distance.distance(t, locs[p]).km

        for i, th in enumerate(distance_thresholds):
            if dist <= th:
                dst_counts[i] += 1

        avg_dist += dist

    avg_dist /= preds.shape[0]
    print(f"Avg. Distance: {avg_dist}")
    for i, cnt in enumerate(dst_counts):
        print(f"Accuracy at {distance_thresholds[i]}: {cnt/preds.shape[0]}")
        

        


