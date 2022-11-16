from matplotlib import pyplot as plt
import torch
from torchvision.transforms import ToTensor, Resize, Compose, ConvertImageDtype
from torchvision.transforms import InterpolationMode
import argparse

# Imports from this project
import sys
sys.path.insert(0, "../src")
from datasets import SilhouetteDataset
from mappings import int2label, get_image, get_silhouette_simple

# Parse command-line
parser = argparse.ArgumentParser(description="Demo for running net on silhouette images.")
parser.add_argument("-b", "--batchsize", type=int, default=4)
parser.add_argument("--countclasses", action="store_true", help="Whether to print number of images per class.")
args = parser.parse_args()

# Set up datasets
images = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=get_image,
    transform=Compose([ToTensor(), Resize((224, 224))])
)
silhouettes = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=get_silhouette_simple,
    transform=Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.NEAREST), # nearest neighbor interpolation conserves narrow structures better
        ConvertImageDtype(torch.float32)
    ])
)

if args.countclasses:
    class_counts = images._count_classes()
    print("Total number of images:", len(images))
    print(f"Class occurences ({len(class_counts)} classes):")
    for c in class_counts:
        print(c.ljust(20, " "), class_counts[c])

from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
weights = ResNet50_Weights.IMAGENET1K_V2
net = resnet50(weights=weights)

b = args.batchsize
loadimgs = DataLoader(images, batch_size=b, shuffle=False)
loadsils = DataLoader(silhouettes, batch_size=b, shuffle=False)

img, lbl = next(iter(loadimgs))
sil, _ = next(iter(loadsils))
p1 = net(img)
p2 = net(1 - sil)
for i in range(b):
    c1 = weights.meta["categories"][p1[i].argmax()]
    c2 = weights.meta["categories"][p2[i].argmax()]
    plt.subplot(2, b, i+1)
    plt.imshow(img[i].permute(1,2,0).numpy())
    plt.title(f"{int2label[lbl[i].item()]}, predicted: {c1}")
    plt.subplot(2, b, b+i+1)
    plt.imshow(sil[i].permute(1,2,0).numpy(), cmap="gray")
    plt.title(f"{int2label[lbl[i].item()]}, predicted: {c2}")
plt.show()


