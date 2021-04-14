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
python examples/assimilate_double_gyre.py
```

## 2. Data requirements

See the [dedicated documentation file](docs/data_requirements.md).

## 3. Specific requirements to test the program on a double gyre simulation

These are the steps to follow to reproduce the results in the initial paper for the GMD journal.
See the [dedicated documentation file](docs/double_gyre.md)

## 4. Start the simulation

Copy one of the examples scripts in the root folder and set the parameters to what suits your situation.
The parameters to the `run_assimilator` function are listed in the `src/run_assimilator.py`

The program will create a `data_.../` folder in the main data folder, and an `output_.../` folder in the outputs folder. If these folders already exist, the simulation will not start in order to not erase existing data.

Graphs and metrics csv logs are periodically output of the simulation and written in the output folder.

The netCDF data in the `data_.../` folder is only written at the end of the model, or if you interrupt it with Ctrl+C.

# Appendices

## A. Notebooks used to generate figures for the initial paper

These notebooks are in the `analysis/` folder.
