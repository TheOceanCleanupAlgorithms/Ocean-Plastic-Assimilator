# Data Assimilation of Plastics Concentration in a Lagrangian Model - Algorithms

## Version 1

This repository contains code to perform data assimilation on a dispersion of particles in a controlled environment.
For now the goal is to prove such a method allows to correct a forecast accurately. In time, it will be used on dispersion data in real-world flow fields.

The following steps describe how to use the code to apply the assimilation method on a double gyre.

## 1. Main variables

All parameters and constants are defined in the `sim_vars.py` file. You can change them to perform a new simulation.
Please be careful after changing constants. Parameters changes are planned and should not cause too many problems.

## 2. Runtime requirements

The required packages to perform a simulations are available in the `env.txt` file.
You can install them easily with `conda` (install conda first with anaconda or miniconda). Please not you may need to add the conda-forge channels to your config:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Then, create the environment with one simple command:

`conda create --name <YOUR_ENV_NAME> --file env.txt`

And then, activate it:

`conda activate <YOUR_ENV_NAME`

That's it, your good to go!

## 3. Data requirements

This program requires two datasets of dispersion data. They are to be put in the folder which the constant `MAIN_DIR_PATH` points to.

They have to be netCDF4 datasets with the following structure:

```
<class 'netCDF4._netCDF4.Dataset'>
root group (NETCDF4 data model, file format HDF5):
    dimensions(sizes): x(25000), time(2001)
    variables(dimensions): int32 id(x), float64 weight(x), float64 time(time), float64 lon(x,time), float64 lat(x,time)
    groups:
```

where the size of dimension x is equal to constant `NB_PARTS` (25000) and the size of dimension time is equal to constant `TIMES` (2001).

- The first one of the two is `parts_double_gyre_ref_eps_{epsilon}_A_{A}.nc` and contains the dispersion data of the reference situation. If you change the flow field parameters A and epsilon, you will have another dispersion dataset and it has to be named this way.

- The second one is `parts_double_gyre_useful_{NB_PARTS}_as_{altSeed}.nc` and contains the dispersion data which will undergo data assimilation. It is is called useful because it's not the one that will be modified per se, but copied in the simulation data folder. Its copy will then be attributed particle weights and will have its values changed by data assimilation.

**/!\\ Be careful to dispersion seeds**

You can see that the useful dataset has an altSeed variable. Datasets with different versions of altSeed have different initial particles. It's been implemented in order to be sure the particles in the forecast and in the reference situation are not the exact same ones. In the case they were, you would see immediate convergence, but this is not something we want because we want different simulations.

> As such, please make sure that `altSeed` is at least equal to 3. The results in the paper are generated using `altSeed = 3`.

### 3.a (RECOMMENDED) Using the data generated for the paper.

All the required files are available at this address : [Zenodo archive : 10.5281/zenodo.4278041](http://doi.org/10.5281/zenodo.4278041)

Just download them and put them in the data folder `MAIN_DIR_PATH` you have defined in `sim_vars.py`.

Set `altSeed = 3` after.

### 3.b Generating new dispersion data and experiment with it

This repository contains the code to generate your own two sets of dispersion data based on the analytical double gyre flow field presented in the manuscript. You just have to follow these steps:

- Define the parameters `A`, `epsilon` and `altSeed` you want to use for the dispersion to generate.
- Execute the following command:
  `python src/double_gyre/create_double_gyre.py`
- Rename the generated file in order to make either a reference data one or a future forecast one.

**Seed requirements: If you're generating a reference situation, please use altSeed < 2, and altSeed > 2 for a forecast situation**

## 4. Start the simulation

Run the command `python init_and_run_simulation.py` in the root folder.

It will create a `data_.../` folder in the main data folder, and an `output_.../` folder in the outputs folder. If these folders already exist, the simulation will not start in order to not erase existing data.

Graphs and metrics csv logs are periodically output of the simulation and written in the output folder. You can change the period in the parameters in `sim_vars.py`

The netCDF data in the `data_.../` folder is only written at the end of the model, or if you interrupt it with Ctrl+C.

# Annexes

## B. Notebooks used to generate figures

These notebooks are in the `analysis/` folder.
