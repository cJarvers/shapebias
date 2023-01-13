# shapebias - Experiments on shape processing in deep networks


## Dependencies and Installation

The project requires several dependencies, not all of which are available as packages.
Packaged dependencies are listed in the conda `environment.yml` file. Install them with:

```
conda env create -f environment.yml
```

This should install `pytorch`, `torchvision` and dependencies.

Next, install [`rsatoolbox`](https://rsatoolbox.readthedocs.io/en/latest/index.html) using pip:

```
pip install rsatoolbox
```

The following packages implement some of the networks that were evaluated. Install them with pip:

```
pip install git+https://github.com/wielandbrendel/bag-of-local-features-models.git
pip install git+https://github.com/dicarlolab/CORnet
```

## Usage

To reproduce the results from the paper, run the following scripts from the `experiments` folder:
- `classify.py` for evaluating accuracies
- `rsa_analysis.py` for RSAs

You can choose the networks and image types to evaluate via command line arguments. For example,

```
python rsa_analysis.py --crop --datasets image fg silhouette frankenstein serrated --network vgg19
```

will run the classification analysis for VGG-19 on the cropped image versions.

The result data will be saved to the `results` folder. To generate the plots show in the paper, use
the scripts `plot_accuracy.py` and `plot_rsa.py`. See the script `plotall.sh` for a complete example.