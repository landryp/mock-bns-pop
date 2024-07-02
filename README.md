##### mock-bns-pop
### simulate a binary neutron star merger population and calculate Fisher matrix parameter uncertainties
philippe landry (pgjlandry@gmail.com) 02/2024

*this repository contains a python script for generating a mock population of binary neutron star mergers and estimating (via gwbench) the parameter uncertainties that would be recovered if observed with a specified gravitational-wave detector network*

### Notebooks

*these notebooks offer a demonstration of the analysis and showcase the key results*

FisherMatrixBNSpop.ipynb *# simulated binary neutron star merger population with Fisher matrix parameter uncertainties*

### Utilities

*the src/ directory contains the utilities needed for the analysis*

utils.py *# utilities for mock population generation*

### Scripts

*these scripts are used for production analyses*

FisherMatrixBNSpop.py inifile *# simulate binary neutron star merger population with Fisher matrix parameter uncertainties, based on .ini file input*
