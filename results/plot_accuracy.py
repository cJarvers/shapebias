import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, "../src")
from loadnetworks import netnamenice
from fdr import fdrcorrection

parser = argparse.ArgumentParser(description="Read and plot accuracy results.")
parser.add_argument("-f", "--filename", type=str, required=True, help="Path to file with accuracies")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to file to save figure in.")
parser.add_argument("--show", action="store_true", help="Whether to show resulting figure.")
parser.add_argument("--fdr", type=float, default=0.05, help="Value at which to control false discovery rate.")
cmdargs = parser.parse_args()

# load RSA results
saved_state = torch.load(cmdargs.filename)
accuracies = saved_state["accuracies"]

# recover parameters of RSA run that are needed for plotting
args = saved_state["commandline"]

# settings for plotting
plt.rcParams.update({
    'font.size': 15,
    'figure.figsize': (8, 5),
    'axes.spines.right': False,
    'axes.spines.top': False
})

# plot results
xs = np.arange(1, len(args.datasets)*2+1, 2)
heights = np.array([accuracies[dset]["acc"] for dset in accuracies])
pvals = np.array([accuracies[dset]["p-value"] for dset in accuracies])
significant = fdrcorrection(pvals, Q=cmdargs.fdr) # control false discovery rate
plt.bar(xs, height=heights)
#plt.boxplot([accuracies[dset]["perm_accs"] for dset in accuracies], positions=xs+1)
plt.violinplot([accuracies[dset]["perm_accs"] for dset in accuracies], positions=xs+1, showmeans=True, widths=1.0)
plt.scatter(xs[significant], heights[significant] + 0.05, marker="*", c="black")
plt.gca().set_xticks(xs+0.5, args.datasets)
plt.ylabel("accuracy")
plt.ylim(top=0.75, bottom=0.0)
plt.title(f"Accuracy of {netnamenice(args.network)}")
plt.tight_layout()
# save figure to file
plt.savefig(cmdargs.output)
if cmdargs.show:
    plt.show()