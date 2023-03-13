import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys
sys.path.insert(0, "../src")
from loadnetworks import netnamenice, vgg19_nicenames, vit_nicenames, alexnet_nicenames
from fdr import fdrcorrection

parser = argparse.ArgumentParser(description="Read RSA results and plot similarities.")
parser.add_argument("-f", "--filename", type=str, required=True, help="Path to file with RSA results")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to file to save figure in.")
parser.add_argument("--show", action="store_true", help="Whether to show resulting figure.")
parser.add_argument("--labelrotation", type=int, help="Rotate xtick labels in plot", default=0)
parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
cmdargs = parser.parse_args()

# load RSA results
saved_state = torch.load(cmdargs.filename)
comparisons = saved_state["comparisons"]

# recover parameters of RSA run that are needed for plotting
args = saved_state["commandline"]
all_datasets = args.comparisons + [args.baseline]
all_layers = args.layers + ["image", "label"]
comparison_layers = ["image"] + args.layers
if args.method == "fixed":
    method_string = r'rank-correlation $(\rho_a)$'
elif args.method == "weighted":
    method_string = r'linear correlation $(r)$'

# VGG and ViT layer names are not nice - load nicer ones if necessary
if args.network == "vgg19":
    layernames = ["image"] + vgg19_nicenames
elif args.network == "vit":
    layernames = ["image"] + vit_nicenames
elif args.network == "alexnet":
    layernames = ["image"] + alexnet_nicenames
else:
    layernames = list(comparisons.keys())

# Plot all comparison results in custom bar plot
xs = np.array([i*(len(all_datasets))+j+1 for i in range(len(comparison_layers)) for j in range(len(args.comparisons))])
xticks = [i*len(all_datasets)+0.5+len(args.comparisons)/2 for i in range(len(comparison_layers))]
heights = np.concatenate([comparisons[layer].get_means() for layer in comparison_layers])
lower_error = heights - np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[0] for layer in comparison_layers])
upper_error = np.concatenate([comparisons[layer].get_ci(0.95, test_type="bootstrap")[1] for layer in comparison_layers]) - heights
pvals = np.concatenate([c.test_zero(test_type="bootstrap") for c in comparisons.values()])
significant = fdrcorrection(pvals, Q=cmdargs.fdr) # control FDR
colorseq = [mpl.colors.to_rgb(mpl.colors.TABLEAU_COLORS[k]) for k in mpl.colors.TABLEAU_COLORS]
colors = [colorseq[i] for _ in range(len(comparisons.keys())) for i in range(len(args.comparisons))]
legend = [mpl.patches.Patch(color=colorseq[i], label=dset) for i, dset in enumerate(args.comparisons)]
plt.bar(x=xs, height=heights, yerr=[lower_error, upper_error], color=colors)
plt.scatter(x=xs[significant], y=heights[significant]+0.1, marker="*", color="black")
plt.xticks(xticks, labels=layernames, rotation=cmdargs.labelrotation)
plt.legend(handles=legend)
plt.title(f"Similarity to {args.baseline} representations in {netnamenice(args.network)}")
plt.ylabel(method_string)
plt.tight_layout()
# save figure to file
plt.savefig(cmdargs.output)
if cmdargs.show:
    plt.show()