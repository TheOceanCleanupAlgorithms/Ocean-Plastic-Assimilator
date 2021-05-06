# Double gyre experiment

In order to experiment with the assimilation scheme in a controlled environment, we can generate different double gyre flow fields, then disperse particles in them and try to assimilate observations sampled in one simulation into another.

This is the method used for the initial paper to present this scheme, available at : NOT RELEASED YET

Use the following to generate two dispersion simulations with the flow field defined with the parameters you want, and then input them into the assimilator with the `observation_type` defined as `from_simulation`

## (RECOMMENDED) Using the data generated for the paper.	

All the required files are available using the following DOI: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4740138.svg)](https://doi.org/10.5281/zenodo.4740138)


You can simply download the files and use them as input of the assimilator using the examples available in the `examples/` folder or the `examples_index.ipynb` notebook (recommended). The notebook will also help you easily download the files in the right directory.

## Generating a double gyre simulation

- You need [opendrift](https://opendrift.github.io/) to be installed.
- Define the parameters you want inside the `src/double_gyre/create_doublegyre.py` file. The roles of A, epsilon and omega are explained inside the paper, or at [this address](https://shaddenlab.berkeley.edu/uploads/LCS-tutorial/examples.html).
- Execute `python create_double_gyre.py`


#### /!\ IMPORTANT /!\ 
If you want to assimilate observations sampled from a simulation into one generated with the exact same flow field and number of particles, make sure to generate them with different `altSeed`. Otherwise they will be exactly similar.