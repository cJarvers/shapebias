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
from datasets import SilhouetteDataset
from mappings import get_silhouette_simple, get_silhouette_bbox
from loadnetworks import load_resnet50

# Parse command-line
parser = argparse.ArgumentParser(description="Demo for running RSA on network layers with silhouette images.")
parser.add_argument("-l", "--layers", type=str, nargs="+", help="Names of layers to use for RSA.", required=True)
parser.add_argument("-b", "--batchsize", type=int, default=32)
parser.add_argument("--crop", action="store_true", help="Whether to restrict images / silhouettes to scaled bounding box.")
args = parser.parse_args()

# Set up datasets
if args.crop:
    mapping = get_silhouette_bbox
else:
    mapping = get_silhouette_simple

silhouettes = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=mapping,
    transform=Compose([
        ToTensor(),
        Resize((224, 224), interpolation=InterpolationMode.NEAREST), # nearest neighbor interpolation conserves narrow structures better
        ConvertImageDtype(torch.float32)
    ])
)

net = load_resnet50(args.layers)

b = args.batchsize
loader = DataLoader(silhouettes, batch_size=b, shuffle=False)
activations = {layer: [] for layer in args.layers}
activations["image"] = []
activations["label"] = []
classes = []
flat = torch.nn.Flatten()
for img, lbl in loader:
    classes.append(lbl)
    acts = net(img)
    activations["image"].append(flat(img))
    activations["label"].append(torch.nn.functional.one_hot(lbl, num_classes=21))
    for layer in args.layers:
        activations[layer].append(flat(acts[layer]))

classes = torch.cat(classes).numpy()
#classes = [int2label[c] for c in classes]
activations = {layer: torch.cat(activation).numpy() for layer, activation in activations.items()}

import rsatoolbox
rsa_datasets = {
    layer: rsatoolbox.data.Dataset(
        activation,
        obs_descriptors={"classes": classes, "image": list(range(len(silhouettes)))},
        channel_descriptors={"unit": list(range(activation.shape[1]))}
    ) for layer, activation in activations.items()
}
for _, dataset in rsa_datasets.items():
    dataset.sort_by("classes")

rdms = {
    layer: rsatoolbox.rdm.calc_rdm(
        rsa_datasets[layer],
        method = "euclidean",
    ) for layer in activations
}

for (layer, rdm) in rdms.items():
    rsatoolbox.vis.show_rdm(
        rdm,
        rdm_descriptor=layer,
        #pattern_descriptor="classes",
        #num_pattern_groups=20,
        show_colorbar="panel"
    )
plt.show()



