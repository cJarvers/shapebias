# Performs RSA on representations of a network for natural images,
# images with foreground masked, and images with background masked.
# The goal is to assess how much representations reflect fore- or background.
import argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import rsatoolbox
import torch
from torch.utils.data import DataLoader
torch.set_grad_enabled(False)

# Imports from this project
import sys
sys.path.insert(0, "../src")
from datasets import SilhouetteDataset
from mappings import get_image, get_image_fg, get_image_bg, get_image_bbox, get_image_fg_bbox, get_image_bg_bbox
from loadnetworks import load_resnet50

parser = argparse.ArgumentParser(description="Runs RSA to assess how much representations are driven by fore- and background.")
parser.add_argument("-l", "--layers", type=str, nargs="+", help="Names of layers to use for RSA.", required=True)
parser.add_argument("-b", "--batchsize", type=int, default=32)
parser.add_argument("--crop", action="store_true", help="Whether to restrict images / silhouettes to scaled bounding box.")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

#################
# Load datasets #
#################
if args.verbose:
    print("Loading datasets ...")
if args.crop:
    img_mapping = get_image_bbox
    fg_mapping = get_image_fg_bbox
    bg_mapping = get_image_bg_bbox
else:
    img_mapping = get_image
    fg_mapping = get_image_fg
    bg_mapping = get_image_bg

images = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=img_mapping,
)
foregrounds = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=fg_mapping,
    sinds=images.sinds, dinds=images.dinds
)
backgrounds = SilhouetteDataset("../data", image_set="val",
    filters=["single", "occluded", "truncated"],
    mapping=bg_mapping,
    sinds=images.sinds, dinds=images.dinds
)
imloader = DataLoader(images, batch_size=args.batchsize)
fgloader = DataLoader(foregrounds, batch_size=args.batchsize)
bgloader = DataLoader(backgrounds, batch_size=args.batchsize)
datasets = {
    "images": images,
    "foregrounds": foregrounds,
    "backgrounds": backgrounds
}

# Load network
net = load_resnet50(args.layers)

###################
# Get activations #
###################
if args.verbose:
    print("Computing activations ...")
# Preprare dictionary to log activations to
args.layers = args.layers + ["image", "label"]
activations = {
    dset: {
        layer: [] for layer in args.layers
    } for dset in datasets.keys()
}
classes = []
flat = torch.nn.Flatten()
# Record activations for images in dataset
for (img, lbl), (fg, _), (bg, _) in zip(imloader, fgloader, bgloader):
    classes.append(lbl.numpy())
    for k, x in {"images": img, "foregrounds": fg, "backgrounds": bg}.items():
        activations[k]["image"].append(flat(x).numpy())
        activations[k]["label"].append(torch.nn.functional.one_hot(lbl, num_classes=21).numpy())
        outputs = net(x)
        for layer, o in outputs.items():
            activations[k][layer].append(flat(o).numpy())

# Postprocess activations
for dset in datasets.keys():
    for layer in args.layers:
        activations[dset][layer] = np.concatenate(activations[dset][layer])
classes = np.concatenate(classes)

########################################
# Representational similarity analysis #
########################################
if args.verbose:
    print("Performing RSA ...")
# Format data for rsatoolbox
image_index = list(range(len(images))) # descriptor for image identity
rsa_datasets = {}
for dset in datasets.keys():
    rsa_datasets[dset] = {}
    for layer in args.layers:
        activation = activations[dset][layer]
        rsa_datasets[dset][layer] =  rsatoolbox.data.Dataset(
            activation,
            obs_descriptors={"classes": classes, "image": image_index},
            channel_descriptors={"unit": list(range(activation.shape[1]))}
        )
        rsa_datasets[dset][layer].sort_by("classes")

# Potential intermediate steps:
# - Reduce dimensionality by PCA (e.g., as in https://www.biorxiv.org/content/10.1101/2020.05.07.082743v1)
#   This requires a separate dataset to perform the RSA on.
#   Since we are interested in how the network responds on different input data domains, this might backfire.
#   For example, a component that has high variation on silhouettes does not have high variation on natural images and is discarded.
#   Thus, we leave out this step.

# Generate RDMs
rdms = {
    dset: {
        layer: rsatoolbox.rdm.calc_rdm(
            rsa_datasets[dset][layer],
            method = "euclidean",
        ) for layer in args.layers
    } for dset in datasets.keys()
}

# Plot RDMs in one view
if False:
    rows = len(list(datasets.keys()))
    cols = len(args.layers)
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    for (r, dset) in enumerate(datasets.keys()):
        subfig = subfigs[r]
        subfig.suptitle(dset)
        axs = subfig.subplots(nrows=1, ncols=cols)
        # leftmost plot is image RDM
        rsatoolbox.vis.rdm_plot.show_rdm_panel(
            rdms[dset]["image"],
            ax=axs[0],
            rdm_descriptor="image",
        )
        # rightmost plot is label RDM
        rsatoolbox.vis.rdm_plot.show_rdm_panel(
            rdms[dset]["label"],
            ax=axs[cols-1],
            rdm_descriptor="class",
        )
        # RDMs of other layers shown in order
        for (c, layer) in enumerate(args.layers[:-2]):
            rsatoolbox.vis.rdm_plot.show_rdm_panel(
                rdms[dset][layer],
                ax=axs[c+1],
                rdm_descriptor=layer,
            )
    plt.show()

# Compare RDMs
# - For each "images" RDM, compare "foregrounds" and "backgrounds" RDMs from the same layer.
#   This 
comparisons = {}
for layer in args.layers[:-2]:
    fg_model = rsatoolbox.model.ModelWeighted("foreground", rdms["foregrounds"][layer]) # Weighted models - by reweighting features, comparison could be more selective to units that actually reflect shape.
    bg_model = rsatoolbox.model.ModelWeighted("background", rdms["backgrounds"][layer]) # However, this requires differentiable dissimilarity measures, which are less robust.
    comparisons[layer] = rsatoolbox.inference.bootstrap_crossval(
        models=[fg_model, bg_model], data=rdms["images"][layer],
        method="corr", # Default "cosine" leads to results near 1 all the time. "corr" seems to be more informative
        #fitter=rsatoolbox.model.fitter.fit_optimize_positive,
        k_rdm=1, # boot_type="pattern" # seems to lead to an error unless also used with k_rdm=1 / when used with fixed models
    )

# Plot all comparison results in custom bar plot
xs = [i*(len(datasets.keys()))+j+1 for i in range(len(comparisons.keys())) for j in range(len(datasets.keys())-1)]
xticks = [i*len(datasets.keys())+0.5+(len(datasets.keys())-1)/2 for i in range(len(comparisons.keys()))]
heights = np.concatenate([comparisons[layer].get_means() for layer in comparisons.keys()])
lower_error = heights - np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[0] for layer in comparisons.keys()])
upper_error = np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[1] for layer in comparisons.keys()]) - heights
colors = [mpl.color_sequences['tab10'][i] for _ in range(len(comparisons.keys())) for i in range(len(datasets.keys())-1)]
legend = [mpl.patches.Patch(color=mpl.color_sequences['tab10'][i], label=dset) for i, dset in enumerate(list(datasets.keys())[1:])]
plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
plt.xticks(xticks, labels=list(comparisons.keys()))
plt.legend(handles=legend)
plt.show()