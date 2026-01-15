import numpy as np
import pandas as pd
import os
import time

'''
Overview:
For each of our two datasets, MMLU-Pro and Non-MMLU-Pro, we will
1. Create the full M1 + M2 splits. (i.e., nfobs=None_p=1.0). Here, M1 means historical data, and M2 means test models.
2. Also look at variants where only {50, 200, 800} rows of M1 are fully observed, with MCAR probability 0.1.
3. Also look at variants where 0 rows of M1 are fully observed, with MCAR probability {0.01, 0.001, 0.0001, 0.00001}.
4. We will pre-sample the above settings for re-use across experiments to ensure reproducibility + fairness.
5. For all experiments, M2 will be ~2K models, left untouched from data scraping.
'''
# relevant settings + missingness parameters (n_full_obs, mcar_obs_prob)
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
MISSINGNESS_SETTINGS = [(50, 0.1), (200, 0.1), (800, 0.1), (0, 0.01), (0, 0.001), (0.0001), (0.00001)]

# how many settings do we have total?
N_SETTINGS = len(DATASETS) * len(MISSINGNESS_SETTINGS)

# counter for the number of settings, total number of settings
counter = 0

# load in the full M1 for each dataset
for dataset in DATASETS:
    
    # load in the M1 as a .csv file + get its true dimensions - also just save a copy as the fully-observed version.
    M1 = pd.read_csv(f"data/processed/{dataset}/M1.csv"); M1.to_csv(f"data/processed/{dataset}/M1_nfobs=None_p=1.0.csv", index=False)
    M1 = M1.astype({col : float for col in M1.columns[3:]})
    n_models, n_questions = M1.iloc[:,3:].shape
    
    # go thru our missingness settings
    for missingness_setting in MISSINGNESS_SETTINGS:

        # unpack the n_full_obs, mcar_obs_prob
        n_full_obs, mcar_obs_prob = missingness_setting
            
        # start a timer
        start = time.time()
        
        # set a seed for reproducibility
        np.random.seed(counter)
        
        # create a matrix of (n_models, n_questions) of all zeroes
        obs_mask = np.zeros(shape=(n_models, n_questions), dtype=bool)

        # which indices are going to be fully observed?
        fully_obs_idxs = np.random.choice(M1.index, replace=False, size=n_full_obs)
        fully_obs_indics = np.zeros(shape=n_models, dtype=bool)
        fully_obs_indics[fully_obs_idxs] = True
        obs_mask[fully_obs_indics,:] = True

        # of the remaining missing entries, let's do MCAR
        obs_mask[~fully_obs_indics] = np.random.binomial(
            n=1, p=mcar_obs_prob, size=obs_mask[~fully_obs_indics].shape).astype(bool)

        # get the entries in M1, nan them out
        M1_vals_censored = M1.iloc[:,3:].to_numpy()
        M1_vals_censored[~obs_mask] = np.nan
        
        # create a duplicate + replace the entries
        M1_censored = M1.copy(deep=True)
        M1_censored.iloc[:,3:] = M1_vals_censored
        
        # save our file to a .csv
        M1_censored.to_csv(
            f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv", index=False)
        
        # increment our counter at the end
        counter += 1
        
        # end our timer
        end = time.time()
        time_elapsed = end - start
        
        # status update + diagnostics
        os.system("clear")
        th_obs_rate = (n_full_obs + (n_models - n_full_obs) * mcar_obs_prob) / n_models
        ac_obs_rate = 1.0 - M1_censored.iloc[:,3:].isna().mean().mean()
        print(f"Finished {counter} of {N_SETTINGS} settings.")
        print(f"\t- Last setting finished in {time_elapsed:.3f} seconds.")
        print(f"\t- Dataset: {dataset}; # Fully Observed: {n_full_obs}; MCAR Obs. Prob: {mcar_obs_prob}.")
        print(f"\t- Theoretical Obs. Rate: {th_obs_rate:.3f}; Actual Obs. Rate: {ac_obs_rate:.3f}.")