from matplotlib import pyplot as plt
import torch
torch.set_grad_enabled(False)
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose, ConvertImageDtype
from torchvision.transforms import InterpolationMode
import argparse

# Imports from this project
import sys
sys.path.insert(0, "../src")
from datasets import loaddataset
from loadnetworks import loadnetwork
from mappings import int2label

# Parse command-line
parser = argparse.ArgumentParser(description="Demo for running net on silhouette images.")
parser.add_argument("-b", "--batchsize", type=int, default=4)
parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets / image types to use as inputs.")
parser.add_argument("--countclasses", action="store_true", help="Whether to print number of images per class.")
parser.add_argument("--set", type=str, default="val", help="PascalVOC image set to use (e.g., 'val').")
args = parser.parse_args()

# Set up datasets
datasets = {}
for imgtype in args.datasets:
    datasets[imgtype], sinds, dinds = loaddataset(imgtype, image_set=args.set)
loaders = {imgtype: DataLoader(dset, batch_size=args.batchsize) for imgtype, dset in datasets.items()}

if args.countclasses:
    dset = datasets[args.datasets[0]]
    class_counts = dset._count_classes()
    print("Total number of images:", len(dset))
    print(f"Class occurences ({len(class_counts)} classes):")
    for c in class_counts:
        print(c.ljust(20, " "), class_counts[c])

# get network
#net = loadnetwork("resnet50", layers=None)

# plot results
fig, axs = plt.subplots(len(args.datasets), args.batchsize)
for i, dset in enumerate(loaders):
    img, lbl = next(iter(loaders[dset]))
    #p = net(img)
    for j in range(args.batchsize):
        axs[i, j].imshow(img[j].permute(1,2,0).numpy())
        # c = p.argmax(dim=1)
        # axs[i,j].title(f"{int2label[lbl[j].item()]}, predicted: {c}")
# set row labels
for ax, dset in zip(axs[:, 0], args.datasets):
    ax.set_ylabel(dset, rotation=0, size="large")
#fig.tight_layout()
plt.show()


