# Demo for how to classify images / silhouettes. Since images are from
# PascalVOC originally, they have twenty classes. Most networks are trained
# on ImageNet with 1000 classes, so the predictions have to be mapped
# onto the 20 PascalVOC categories (and one "background" class).
import numpy as np
import torch
torch.set_grad_enabled(False)
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor, Resize, Compose
import sys
sys.path.insert(0, "../src")
from datasets import SilhouetteDataset
from mappings import get_image_bbox, get_image_fg_bbox, get_image_bg_bbox, get_silhouette_bbox
from helpers.imagenet_synsets import imagenet2voc

# load data
images = SilhouetteDataset("../data", image_set="trainval",
    filters=["single", "occluded", "truncated"],
    mapping=get_image_bbox
)
foregrounds = SilhouetteDataset("../data", image_set="trainval",
    filters=["single", "occluded", "truncated"],
    mapping=get_image_fg_bbox
)
backgrounds = SilhouetteDataset("../data", image_set="trainval",
    filters=["single", "occluded", "truncated"],
    mapping=get_image_bg_bbox
)
silhouettes = SilhouetteDataset("../data", image_set="trainval",
    filters=["single", "occluded", "truncated"],
    mapping=get_silhouette_bbox
)
imloader = DataLoader(images, batch_size=32)
fgloader = DataLoader(foregrounds, batch_size=32)
bgloader = DataLoader(backgrounds, batch_size=32)
shloader = DataLoader(silhouettes, batch_size=32)

# set up network
weights = ResNet50_Weights.IMAGENET1K_V2
net = resnet50(weights=weights)

total = 0
correct = {"image     ": 0, "foreground": 0, "background": 0, "silhouette": 0}
for (img, lbl), (fg, _), (bg, _), (sil, _) in zip(imloader, fgloader, bgloader, shloader):
    for k, x in {"image     ": img, "foreground": fg, "background": bg, "silhouette": sil}.items():
        out = net(x)
        pred = imagenet2voc[out.argmax(dim=1).numpy()]
        correct[k] += np.sum(pred == lbl.numpy())
    total += lbl.size(0)
print("Total number of images: ", total)
print("Correct responses / accuracies:")
for k, c in correct.items():
    print(k, "\t", c, "correct", f"\t({c / total * 100}%)")