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
from mappings import get_image, get_silhouette_simple, get_image_bbox, get_silhouette_bbox
from helpers.imagenet_synsets import imagenet2voc

# load data
images = SilhouetteDataset("../data", image_set="trainval",
    filters=["single", "occluded", "truncated"],
    mapping=get_image_bbox,
    transform=Compose([ToTensor(), Resize((224, 224))])
)
loader = DataLoader(images, batch_size=32)

# set up network
weights = ResNet50_Weights.IMAGENET1K_V2
net = resnet50(weights=weights)

total = 0
correct = 0
for img, lbl in loader:
    out = net(img)
    pred = imagenet2voc[out.argmax(dim=1).numpy()]
    correct += np.sum(pred == lbl.numpy())
    total += pred.size
print("Accuracy: ", correct / total)
    