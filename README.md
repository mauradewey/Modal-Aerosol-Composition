## MCMC parameter optimization for inverse-CCN closure using Köhler theory
For the paper: \
Optimizing CCN predictions through inferred modal aerosol composition – a boreal forest case study \
Rahul Ranjan, Maura Dewey, Liine Heikkinen, Lauri R. Ahonen, Krista Luoma, Paul Bowen, Tuukka Petäjä, Annica M. L. Ekman, Daniel G. Partridge and Ilona Riipinen \
Accepted at ACP

Requirements to run the optimization can be found in requirements.yml \
The main code files are main_dask.py and  run_mcmc.py \
To run an mcmc optimization, make sure the correct model is chosen in main_dask.py, then submit on HPC with submit_mcmc_dask.sh. The statistics for the optimized parameters are calculated with summarize_mcmc_posteriors.py. \
The notebook to analyse and plot results is explore-mcmc-results.ipynb


