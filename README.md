[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4278048.svg)](https://doi.org/10.5281/zenodo.4278048)

# Ocean Plastic Assimilator - v0.1

This repository contains the code of the Ocean Plastic Assimilator, a program to perform data assimilation on a dispersion of particles.

## 1. Runtime requirements

The required packages to perform a simulations are available in the `environment.yml` file.
You can install them easily with `conda`. Please note that you will need to add the conda-forge channels to your config:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

## 1.1 Get Started

```
conda env create -f environment.yml
conda activate ocean-plastic-assimilator
```

then use one of the example scripts in `examples/` as a template for your use case.

## 1.2 Run the GMD paper experiments

If you are looking at running the experiments and examples presented in the paper submitted to Geoscientific Model Development, use the examples_index.ipynb notebook to automatically download the required data and start the experiments.

## 2. Data requirements

See the [dedicated documentation file](docs/data_requirements.md).

## 3. Start the simulation

Follow the instructions in the notebook `examples.ipynb` to run the experiments described in the paper, or copy one of the example start scripts in `examples/` in the root directory and start it.

# Appendices

## A. Notebooks used to generate figures for the GMD paper

These notebooks are in the `analysis/` folder.
