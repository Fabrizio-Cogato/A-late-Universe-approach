# A late-Universe approach to the weaving of modern cosmology 
## [F. Cogato, M. Moresco, L. Amati and A. Cimatti] 

This repository contains the main tool implemented to carry out the cosmological analysis described in https://ui.adsabs.harvard.edu/abs/2023arXiv230901375C/abstract

* Analytical_Cosmo_Config.py | Python script defining functions, likelihood, prior and posterior distribution for the following cosmological models:
    - Flat LCDM
    - LCDM
    - Flat wCDM
    - wCDM
    - Flat w0waCDM
    - w0waCDM

* Analytical_Cosmo.py | Python script for running MCMC sampling.

* path/to/
    - data:   folder with data and covariance matrices of BAO, CC and SN. Please note that GRB data are not presented here and will be published in Amati et al. (in prep.).
    - chains: folder with all the Monte Carlo Markov Chains. Note that, due to GitHub limitations, here are just reported the last 500 steps of the full chains. Complete MCMC will be made available upon request to the authors.
    - plots:  folder with corner plots for each MCMC listed in path/to/chains.
    - tables: folder with latex tables for each MCMC listed in path/to/chains.
