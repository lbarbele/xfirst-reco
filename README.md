# xfirst-reco

A machine-learning project for the reconstruction of the depth of first interaction of ultra-high-energy extensive air showers using fluorescence detectors.

# Project structure

+ `models`: contains models already trained
+ `notebooks`: exploratory data analysis and drafts of model implementations
+ `scripts`: python scripts for manipulating data and training models
+ `sd-scripts`: scripts to launch jobs on slurm controllers
+ `xfirst`: library providing functionality to manipulate data from CONEX files, train models, and input/output operations

# Scripts pipeline

1 - `make_datasets.py`: splits library of CONEX files into train, validation, and test datasets; extracts data from CONEX files (longitudinal energy-deposit profiles, primary energy, and $X_\mathrm{first}$ values).

2 - `make_fits.py`: fits a Universal Shower Profile (USP) function to each shower simulated in CONEX (and previously extraced in step one); the fitted data can be used to train regression models; uses `scipy`'s `curve_fit` implementation.

3 - `train_xgb_fit.py`: using XGBoost, trains a gradient-boosting algorithm to compute the depth of first interaction, $X_\mathrm{first}$ from the fitted profiles (generated on step 2).

4 - `train_mlp_fit.py`: uses TensorFlow to build a simple, small multilayer perceptron network (MLP) to compute $X_\mathrm{first}$ from profile fits.

5 - `train_mlp_profile.py`: builds a MLP that is able to perform regression of $X_\mathrm{first}$ on the full profile extracted from CONEX; optionally, can use fit data to increase the speed of convergence and, possibly, optmize results.