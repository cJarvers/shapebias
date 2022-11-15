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

Then install the [model-vs-human toolbox](https://github.com/bethgelab/model-vs-human). On Linux:

```
git clone https://github.com/bethgelab/model-vs-human.git
export MODELVSHUMANDIR=$(pwd)/model-vs-human/
cd model-vs-human
pip install -e .
```

(Note: if the `pip install` step fails, you may need to adjust some of the version numbers in
the file `model-vs-human/setup.cfg`. Some versions are outdated and not accessible anymore.)

## Usage

