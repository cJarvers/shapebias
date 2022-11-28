# Performs RSA on representations of a network given sets of images.
import argparse
import datetime
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
from datasets import loaddataset
from loadnetworks import loadnetwork

parser = argparse.ArgumentParser(description="Runs RSA to assess how much representations are driven by fore- and background.")
parser.add_argument("--baseline", type=str, required=True, help="Image type to use as baseline for RSA (i.e., to compare other types to).")
parser.add_argument("--comparisons", type=str, nargs="+", required=True, help="Image types to use as comparison cases for RSA.")
parser.add_argument("--crop", action="store_true", help="Whether to restrict images / silhouettes to scaled bounding box.")
parser.add_argument("--set", type=str, default="val", help="PascalVOC image set to use (e.g., 'val').")
parser.add_argument("--network", type=str, default="resnet50", help="Network to extract activations from.")
parser.add_argument("-l", "--layers", type=str, nargs="+", help="Names of layers to use for RSA.", default=["default"])
parser.add_argument("-b", "--batchsize", type=int, default=32)
parser.add_argument("--method", type=str, default="fixed", help="How to compare RDMs. Can be 'fixed' (no weighting, use rho-a) or 'weighted' (weighted models, use corr).")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("--show_rdms", action="store_true", help="If true, shows plot of RDMs (and pauses script halfway).")
parser.add_argument("--show_rsa", action="store_true", help="If true, shows plot of RSA (and pauses script).")
parser.add_argument("-d", "--device", type=str, default="cuda")
args = parser.parse_args()

#################
# Load datasets #
#################
if args.verbose:
    print("Loading datasets ...")
baseline_data, sinds, dinds = loaddataset(args.baseline, args.crop, image_set=args.set)
datasets = {}
for imgtype in args.comparisons:
    datasets[imgtype], _, _ = loaddataset(imgtype, args.crop, image_set=args.set, sinds=sinds, dinds=dinds)
all_datasets = args.comparisons + [args.baseline]

baseline_loader = DataLoader(baseline_data, batch_size=args.batchsize)
loaders = {imgtype: DataLoader(dset, batch_size=args.batchsize) for imgtype, dset in datasets.items()}

# Load network
net, args.layers = loadnetwork(args.network, args.layers, device=args.device) # TODO: add cmdline flag for whether weights should be pretrained

###################
# Get activations #
###################
if args.verbose:
    print("Computing activations ...")
# Preprare dictionary to log activations to
all_layers = args.layers + ["image", "label"] # we also want to record "image" and "label" activations
comparison_layers = ["image"] + args.layers # labels are the same for all datasets, so no need to compare
activations = {
    dset: {
        layer: [] for layer in all_layers
    } for dset in all_datasets
}
classes = []
flat = torch.nn.Flatten()
# Record activations for images in baseline dataset
for (img, lbl) in baseline_loader:
    classes.append(lbl.numpy())
    activations[args.baseline]["image"].append(flat(img).numpy())
    activations[args.baseline]["label"].append(torch.nn.functional.one_hot(lbl, num_classes=21).numpy())
    outputs = net(img.to(args.device))
    for layer, o in outputs.items():
        activations[args.baseline][layer].append(flat(o).cpu().numpy())
for dset, loader in loaders.items():
    for img, lbl in loader:
        activations[dset]["image"].append(flat(img).numpy())
        activations[dset]["label"].append(torch.nn.functional.one_hot(lbl, num_classes=21).numpy())
        outputs = net(img.to(args.device))
        for layer, o in outputs.items():
            activations[dset][layer].append(flat(o).cpu().numpy())

# Postprocess activations
for dset in all_datasets:
    for layer in all_layers:
        activations[dset][layer] = np.concatenate(activations[dset][layer])
classes = np.concatenate(classes)

########################################
# Representational similarity analysis #
########################################
if args.verbose:
    print("Performing RSA ...")
    print("    ... creating datasets")
# Format data for rsatoolbox
#image_index = list(range(len(baseline_data))) # descriptor for image identity
rsa_datasets = {}
for dset in all_datasets:
    rsa_datasets[dset] = {}
    for layer in all_layers:
        activation = activations[dset][layer]
        rsa_datasets[dset][layer] = rsatoolbox.data.Dataset(
            activation,
            obs_descriptors={"classes": classes} #, "image": image_index},
            # channel_descriptors={"unit": list(range(activation.shape[1]))}
        )
        rsa_datasets[dset][layer].sort_by("classes")
    del activations[dset]

del activations # free up some space - with too many comparisons, process may exceed memory otherwise

# Potential intermediate steps:
# - Reduce dimensionality by PCA (e.g., as in https://www.biorxiv.org/content/10.1101/2020.05.07.082743v1)
#   This requires a separate dataset to perform the RSA on.
#   Since we are interested in how the network responds on different input data domains, this might backfire.
#   For example, a component that has high variation on silhouettes does not have high variation on natural images and is discarded.
#   Thus, we leave out this step.
if args.verbose:
    print("    ... calculating RDMs")
# Generate RDMs
rdms = {
    dset: {
        layer: rsatoolbox.rdm.calc_rdm(
            rsa_datasets[dset][layer],
            method = "euclidean",
        ) for layer in all_layers
    } for dset in all_datasets
}

# Plot RDMs in one view
if args.show_rdms:
    rows = len(all_datasets)
    cols = len(all_layers)
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(nrows=rows, ncols=1)
    for (r, dset) in enumerate(all_datasets):
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
        for (c, layer) in enumerate(args.layers):
            rsatoolbox.vis.rdm_plot.show_rdm_panel(
                rdms[dset][layer],
                ax=axs[c+1],
                rdm_descriptor=layer,
            )
    plt.show()

# Compare RDMs
# For each baseline RDM, compare all comparison RDMs to it.
if args.verbose:
    print("    ... computing bootstrapped comparisons")
# First, set up models and fitting procedure
if args.method == "fixed":
    # Fixed models can use robust dissimilarity measures like spearmans rho ('rho-a')
    models = {
        layer: [
            rsatoolbox.model.ModelFixed(dset, rdms[dset][layer]) for dset in args.comparisons
        ] for layer in comparison_layers
    }
    fitter = None # fixed models do not need fitting
    method = 'rho-a'
    method_string = r'rank-correlation $(\rho_a)$'
elif args.method == "weighted":
    # Weighted models - by reweighting features, comparison could be more selective to units that actually reflect shape.
    # We only use positive weights, so dimensions can be up- / down-scaled, but not inverted.
    # However, fitting of model weights requires a differential metric (e.g., 'corr'), which are less robust.
    models = {
        layer: [
            rsatoolbox.model.ModelWeighted(dset, rdms[dset][layer]) for dset in args.comparisons
         ] for layer in comparison_layers
    }
    fitter = rsatoolbox.model.fitter.fit_optimize_positive
    method = 'corr' # Default "cosine" leads to results near 1 all the time. "corr" seems to be more informative
    method_string = r'correlation $(\rho)$'

# Do the actual comparisons:
comparisons = {}
for layer in comparison_layers:
    comparisons[layer] = rsatoolbox.inference.bootstrap_crossval(
        models=models[layer], data=rdms[args.baseline][layer],
        method=method, fitter=fitter,
        k_rdm=1, # boot_type="pattern" # seems to lead to an error unless also used with k_rdm=1 / when used with fixed models
    )

if args.verbose:
    print("    ... plotting results")
# Plot all comparison results in custom bar plot
xs = [i*(len(all_datasets))+j+1 for i in range(len(comparison_layers)) for j in range(len(args.comparisons))]
xticks = [i*len(all_datasets)+0.5+len(args.comparisons)/2 for i in range(len(comparison_layers))]
heights = np.concatenate([comparisons[layer].get_means() for layer in comparison_layers])
lower_error = heights - np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[0] for layer in comparison_layers])
upper_error = np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[1] for layer in comparison_layers]) - heights
colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
colors = [colorseq[i] for _ in range(len(comparisons.keys())) for i in range(len(args.comparisons))]
legend = [mpl.patches.Patch(color=mpl.color_sequences['tab10'][i], label=dset) for i, dset in enumerate(args.comparisons)]
plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
plt.xticks(xticks, labels=list(comparisons.keys()))
plt.legend(handles=legend)
plt.title(f"Similarity to {args.baseline} representations in {args.network}")
plt.ylabel(method_string)
# save figure to file
time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
plt.savefig(f"../results/figures/{time}_rsa_{args.baseline}_{args.network}_{args.method}.png")
if args.show_rsa:
    plt.show()