# Demo for how to classify images / silhouettes. Since images are from
# PascalVOC originally, they have twenty classes. Most networks are trained
# on ImageNet with 1000 classes, so the predictions have to be mapped
# onto the 20 PascalVOC categories (and one "background" class).
import argparse
import datetime
import numpy as np
from matplotlib import pyplot as plt
import torch
torch.set_grad_enabled(False)
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, "../src")
from datasets import loaddataset
from loadnetworks import loadnetwork
from helpers.imagenet_synsets import imagenet2voc

# Parse command line
parser = argparse.ArgumentParser(description="Test for classification performance on modified datasets.")
parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Datasets / image types to use as inputs.")
parser.add_argument("--network", type=str, default="resnet50", help="Network to extract activations from.")
parser.add_argument("--set", type=str, default="val", help="PascalVOC image set to use (e.g., 'val').")
parser.add_argument("--crop", action="store_true", help="Whether to restrict images / silhouettes to scaled bounding box.")
parser.add_argument("-b", "--batchsize", type=int, default=32)
args = parser.parse_args()

# load data
datasets = {}
for imgtype in args.datasets:
    datasets[imgtype], sinds, dinds = loaddataset(imgtype, args.crop, image_set=args.set)
loaders = {imgtype: DataLoader(dset, batch_size=args.batchsize) for imgtype, dset in datasets.items()}

# set up network
net, _ = loadnetwork(args.network, layers=None)

# get network predictions on dataset(s)
predictions = {}
for dset, loader in loaders.items():
    predictions[dset] = {
        "imagenet_class": [],
        "prediction": [],
        "label": []
    }
    for img, lbl in loader:
        out = net(img)
        imgnet_class = out.argmax(dim=1).numpy()
        predictions[dset]["imagenet_class"].append(imgnet_class)
        pred = imagenet2voc[imgnet_class]
        predictions[dset]["prediction"].append(pred)
        predictions[dset]["label"].append(lbl.numpy())
# Postprocessing: concatenate tensors / arrays from all batches
for dset in predictions:
    predictions[dset]["imagenet_class"] = np.concatenate(predictions[dset]["imagenet_class"])
    predictions[dset]["prediction"] = np.concatenate(predictions[dset]["prediction"])
    predictions[dset]["label"] = np.concatenate(predictions[dset]["label"])
# Check that all labels are the same across all datasets (i.e., that the same images were used)
for dset1 in predictions:
    for dset2 in predictions:
        if not (predictions[dset1]["label"] == predictions[dset2]["label"]).all():
            raise ValueError(f"Labels not identical! Datasets {dset1} and {dset2} contain different images!")

# Calculate accuracies
accuracies = {}
for dset in predictions:
    total = predictions[dset]["prediction"].size
    correct = (predictions[dset]["prediction"] == predictions[dset]["label"]).sum()
    accuracies[dset] = {"acc": correct / total}

# Estimate random performance by permutation test
def permutationtest(ps, lbls, n=1000):
    results = [np.mean(np.random.permutation(ps) == lbls) for _ in range(n)]
    return results

for dset in accuracies:
    accuracies[dset]["perm_accs"] = permutationtest(
        predictions[dset]["prediction"],
        predictions[dset]["label"],
        n = 10000
    )
    accuracies[dset]["p-value"] = 1 - (accuracies[dset]["acc"] > accuracies[dset]["perm_accs"]).mean()

# Print and plot results
for dset in predictions:
    print(dset.ljust(30), f"{accuracies[dset]['acc']*100:1.3f}%")

# plot results
xs = np.arange(1, len(args.datasets)*2+1, 2)
heights = np.array([accuracies[dset]["acc"] for dset in accuracies])
pvals = np.array([accuracies[dset]["p-value"] for dset in accuracies])
plt.bar(xs, height=heights)
plt.boxplot([accuracies[dset]["perm_accs"] for dset in accuracies], positions=xs+1)
plt.scatter(xs[pvals < 0.05], heights[pvals < 0.05] + 0.1, marker="*")
plt.gca().set_xticks(xs, args.datasets)
plt.ylabel("accuracy")
plt.title(f"Accuracy of {args.network}")
# save figure to file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
plt.savefig(f"../results/figures/{time}_classification_{args.network}.png")
plt.show()