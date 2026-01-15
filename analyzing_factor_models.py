import numpy as np
import pandas as pd

# load in our logs for the factor model cross-validation splits
df = pd.read_csv("logs/factor_model_cv_logs.csv")

# get averages
df_mean = df.groupby(
    ["dataset", "n_full_obs", "mcar_obs_prob", "K", "lmbda"], dropna=False).mean().reset_index()

# what are the argmax settings so far?
best_settings = df_mean.sort_values(by=["val_acc"], ascending=False)\
.groupby(["dataset", "n_full_obs", "mcar_obs_prob"], dropna=False).first().reset_index().iloc[:,:5]

# create a new dataframe to store our 1sd-cutoffs
osd_cutoffs = pd.DataFrame(columns=["dataset", "n_full_obs", "mcar_obs_prob", "osd_cutoff"])

# go thru our point-estimate best settings
for idx in best_settings.index:
    
    # unpack the parameters for this best setting
    dataset, n_full_obs, mcar_obs_prob, K, lmbda = best_settings.loc[idx]
    
    # get the five values of val_acc
    if ~np.isnan(n_full_obs):
        split_vals = df.query(
            f"dataset == '{dataset}' and n_full_obs == {n_full_obs} " +\
            f"and mcar_obs_prob == {mcar_obs_prob} and K == {K} and lmbda == {lmbda}").val_acc.values
    else:
        split_vals = df.query(
            f"dataset == '{dataset}'" +\
            f"and mcar_obs_prob == {mcar_obs_prob} and K == {K} and lmbda == {lmbda}").val_acc.values
    
    # what is the 1sd-rule cutoff?
    osd_cutoff = np.mean(split_vals) - (np.std(split_vals) / np.sqrt(5))
    osd_cutoffs.loc[len(osd_cutoffs.index)] = [dataset, n_full_obs, mcar_obs_prob, osd_cutoff]
    
    
# new dataframe to store the simplest models that achieve the 1-SD cutoff
selected_settings = pd.DataFrame(columns=df.columns)

# get the most parsimonious settings that achieve the osd-cutoff
for idx in osd_cutoffs.index:
    
    # unpack the parameters for this best setting + the 1sd-cutoff
    dataset, n_full_obs, mcar_obs_prob, osd_cutoff = osd_cutoffs.loc[idx]
    
    # get the settings that hit this cutoff
    if ~np.isnan(n_full_obs):
        candidates = df_mean.query(
            f"dataset == '{dataset}' and n_full_obs == {n_full_obs} " +\
            f"and mcar_obs_prob == {mcar_obs_prob} and val_acc > {osd_cutoff}")
    else:
        candidates = df_mean.query(
            f"dataset == '{dataset}'" +\
            f"and mcar_obs_prob == {mcar_obs_prob} and val_acc > {osd_cutoff}")
        
    # get the most parsimonious setting that hit this cutoff
    candidate = candidates.sort_values(by=["K", "lmbda"], ascending=[True, False]).iloc[0:1]
    
    # get the smallest K + biggest lmbda setting that gets me above the osd_cutoff. add to list
    selected_settings = pd.concat(
        [selected_settings, candidate], ignore_index=True).reset_index(drop=True)
    
# save our results
selected_settings.to_csv("logs/factor_model_selected_settings.csv", index=False)