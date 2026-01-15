'''
Purpose of this script:
1. Extract the <post-hoc> best hyperparameter for our baselines and active inference ablations.
2. Compile dataframes with the 100x seeds per budget x data scenario for each of said best variants.
3. Compile dataframes with the 100x seeds per budget x data scenario for uniform + FAQ methods.
4. Create summary dataframes for plotting purposes: {mean_width, coverage, ess_multiplier} x {mean, serr}.
'''
import numpy as np
import pandas as pd
import os

# how many seeds did we have?
N_SEEDS = 100

'''
Remember that we've already concatenated the files accordingly!
'''
# load in baseline results
baseline_df = pd.read_csv("logs/final/baseline_logs.csv")

# load in active-inference ablation results
ablation_df = pd.read_csv("logs/final/active_inference_ablation_logs.csv")

# load in our FAQ results + make sure it's in sorted order
faq_df = pd.read_csv("logs/final/faq_final_logs.csv")\
.sort_values(by=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed"]).reset_index(drop=True)

# extract out the uniform/human sampling + zero estimator results (per seed)
uniform_df = baseline_df.query("policy == 'unif' and f == 'zero'").copy()\
.sort_values(by=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed"]).reset_index(drop=True)


# the 7 columns that define a baseline setting (aside from seed) + the 4 columns that define a data scenario
variant_cols = list(baseline_df.columns[:7])
scenario_cols = list(baseline_df.columns[:4])

# get the best "Historical" baseline setting per setting, then get per seed results for said setting
baseline_means = baseline_df.groupby(variant_cols, dropna=False).mean().reset_index()
best_baseline_settings = baseline_means.loc[
    baseline_means.groupby(scenario_cols, dropna=False)["mean_width"].idxmin(), variant_cols]
best_baseline_df = pd.merge(baseline_df, best_baseline_settings, how="inner", on=variant_cols)\
.sort_values(by=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed"]).reset_index(drop=True)

# the 5 columns that govern a ablation setting (aside from seed) + the 4 columns that define a data scenario
variant_cols = list(ablation_df.columns[:5])
scenario_cols = list(ablation_df.columns[:4])

# get the best ablation setting per setting, then get per seed results for said setting
ablation_means = ablation_df.groupby(variant_cols, dropna=False).mean().reset_index()
best_ablation_settings = ablation_means.loc[
    ablation_means.groupby(scenario_cols, dropna=False)["mean_width"].idxmin(), variant_cols]
best_ablation_df = pd.merge(ablation_df, best_ablation_settings, how="inner", on=variant_cols)\
.sort_values(by=["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed"]).reset_index(drop=True)

# get ESS multipliers for best baselines & faq vs. uniform
best_baseline_df["ess_multiplier"] = (uniform_df.mean_width / best_baseline_df.mean_width) ** 2
faq_df["ess_multiplier"] = (uniform_df.mean_width / faq_df.mean_width) ** 2

# get the ESS multipliers for best ablations (only on the missingness scenarios we did ablation for)
best_ablation_df["ess_multiplier"] = (
    pd.merge(uniform_df, best_ablation_df[scenario_cols + ["seed"]], 
             on=scenario_cols + ["seed"], how="inner").mean_width / best_ablation_df.mean_width) ** 2

# ESS multiplier summaries by data scenario for FAQ, best-baselines, best-ablations

# make sure we're using the right scenario columns
scenario_cols = list(baseline_df.columns[:4])

# FAQ summary
faq_summary = faq_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(scenario_cols, dropna=False).mean().reset_index()
faq_summary["ess_multiplier_serr"] = faq_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(
    scenario_cols, dropna=False).std().reset_index().ess_multiplier / np.sqrt(N_SEEDS)
faq_summary["coverage"] = faq_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().coverage
faq_summary["coverage_serr"] = faq_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).std().reset_index().coverage / np.sqrt(N_SEEDS)
faq_summary["mean_width"] = faq_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().mean_width
faq_summary["mean_width_serr"] = faq_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).std().reset_index().mean_width / np.sqrt(N_SEEDS)

# best baselines summary
best_baseline_summary = best_baseline_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(scenario_cols, dropna=False).mean().reset_index()
best_baseline_summary["ess_multiplier_serr"] = best_baseline_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(
    scenario_cols, dropna=False).std().reset_index().ess_multiplier / np.sqrt(N_SEEDS)
best_baseline_summary["coverage"] = best_baseline_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().coverage
best_baseline_summary["coverage_serr"] = best_baseline_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).std().reset_index().coverage / np.sqrt(N_SEEDS)
best_baseline_summary["mean_width"] = best_baseline_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().mean_width
best_baseline_summary["mean_width_serr"] = best_baseline_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).std().reset_index().mean_width / np.sqrt(N_SEEDS)

# best-ablations summary
best_ablation_summary = best_ablation_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(scenario_cols, dropna=False).mean().reset_index()
best_ablation_summary["ess_multiplier_serr"] = best_ablation_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(
    scenario_cols, dropna=False).std().reset_index().ess_multiplier / np.sqrt(N_SEEDS)
best_ablation_summary["coverage"] = best_ablation_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().coverage
best_ablation_summary["coverage_serr"] = best_ablation_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).std().reset_index().coverage / np.sqrt(N_SEEDS)
best_ablation_summary["mean_width"] = best_ablation_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().mean_width
best_ablation_summary["mean_width_serr"] = best_ablation_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).std().reset_index().mean_width / np.sqrt(N_SEEDS)

# make sure we're using the right scenario columns
scenario_cols = ["dataset", "prop_budget"]

# get uniform results too in same summary format
uniform_df["ess_multiplier"] = 1.0

# best uniform summary
uniform_summary = uniform_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(scenario_cols, dropna=False).mean().reset_index()
uniform_summary["ess_multiplier_serr"] = uniform_df[
    scenario_cols + ["seed"] + ["ess_multiplier"]].groupby(
    scenario_cols, dropna=False).std().reset_index().ess_multiplier / np.sqrt(N_SEEDS)
uniform_summary["coverage"] = uniform_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().coverage
uniform_summary["coverage_serr"] = uniform_df[
    scenario_cols + ["seed"] + ["coverage"]].groupby(
    scenario_cols, dropna=False).std().reset_index().coverage / np.sqrt(N_SEEDS)
uniform_summary["mean_width"] = uniform_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).mean().reset_index().mean_width
uniform_summary["mean_width_serr"] = uniform_df[
    scenario_cols + ["seed"] + ["mean_width"]].groupby(
    scenario_cols, dropna=False).std().reset_index().mean_width / np.sqrt(N_SEEDS)

# save everything - start with the per-seed logs
best_baseline_df.to_csv("logs/final/cleaned/best_baseline_df.csv", index=False)
best_ablation_df.to_csv("logs/final/cleaned/best_ablation_df.csv", index=False)
faq_df.to_csv("logs/final/cleaned/faq_df.csv", index=False)
uniform_df.to_csv("logs/final/cleaned/uniform_df.csv", index=False)

# save everything - make sure we save these summaries, for easy plotting
best_baseline_summary.to_csv("logs/final/cleaned/best_baseline_summary.csv", index=False)
best_ablation_summary.to_csv("logs/final/cleaned/best_ablation_summary.csv", index=False)
faq_summary.to_csv("logs/final/cleaned/faq_summary.csv", index=False)
uniform_summary.to_csv("logs/final/cleaned/uniform_summary.csv", index=False)