import numpy as np
import pandas as pd
import os

# load in our log-files, concatenated together.
logs = pd.read_csv("logs/val/faq_val_logs.csv")

# average results across seeds, sort by mean_width
mean_logs = logs.groupby(
    list(logs.columns[:8]), dropna=False
).mean().reset_index().sort_values(by="mean_width")

# get our best settings for each dataset + missingness condition + prop_budget
best_settings = mean_logs.groupby(
    ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget"], dropna=False
).first().reset_index()

# save our best settings to a .csv
best_settings.to_csv("logs/val/best_settings.csv", index=False)