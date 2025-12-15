## MCMC parameter optimization for inverse-CCN closure using Köhler theory
Please cite as: \
Ranjan, R., Dewey, M., Heikkinen, L., Ahonen, L. R., Luoma, K., Bowen, P., Petäjä, T., Ekman, A. M. L., Partridge, D. G., and Riipinen, I.: Optimizing CCN predictions through inferred modal aerosol composition – a boreal forest case study, Atmos. Chem. Phys., 25, 17275–17300, https://doi.org/10.5194/acp-25-17275-2025, 2025.

Requirements to run the optimization can be found in requirements.yml \
The main code files are main_dask.py and  run_mcmc.py \
To run an mcmc optimization, make sure the correct model is chosen in main_dask.py, then submit on HPC with submit_mcmc_dask.sh. The statistics for the optimized parameters are calculated with summarize_mcmc_posteriors.py. \
The notebook to analyse and plot results is explore-mcmc-results.ipynb \
Chain summary statistics for analysis and plotting are in the results folder.


