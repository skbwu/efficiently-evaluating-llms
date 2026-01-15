import numpy as np
import pandas as pd
import torch
import sys, os, gc
from scipy.stats import norm
from tqdm.autonotebook import tqdm

# let's set a device too
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Baselines:
1. Sampling Policy -- uniform, sqrt(p(1-p)), 2min(p, 1-p)
2. ML Predictor -- 0, or global mean on this question (potentially imputed).

Settings to iterate over:
1. Datasets: MMLU-Pro and Non-MMLU-Pro
2. Missingness settings: (n_full_obs in {50, 200, 800} x mcar_obs_prob=0.1) + 100% fully-obs + (n_full_obs=0 x mcar_obs_prob \in (0.01, 0.001, 0.0001, 0.00001))
3. Budget Proportion Values: 10 evenly-spaced values between 0.025 and 0.25.
4. Uniform Mixing Proportions (only for sqrt and 2min variants): [0.05, 0.25, 0.5, 0.75]
'''
# datasets + missingness settings, also budget_props and sampling policies
dataset = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"][int(sys.argv[1])]
MISSINGNESS_SETTINGS = [(50, 0.1), (200, 0.1), (800, 0.1), (None, 1.0), (0, 0.00001), (0, 0.0001), (0, 0.001), (0, 0.01)]

# do a second command-line argument to split MISSINGNESS_SETTINGS into half.
MISSINGNESS_SETTINGS = MISSINGNESS_SETTINGS[:4] if int(sys.argv[2]) == 0 else MISSINGNESS_SETTINGS[4:]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
POLICIES = ["unif", "sqrt", "2min"]

# Zrnic & Candes, Section 7.2: Practical Sampling Rules
TAUS = [0.05, 0.25, 0.5, 0.75]

# fixed hyperparameters
ALPHA, N_SEEDS = 0.05, 100

# how many settings are we running total for this script?
NUM_SETTINGS = len(MISSINGNESS_SETTINGS) * len(BUDGET_PROPS) * N_SEEDS *( (len(TAUS) * (len(POLICIES) - 1)) + 1 )

# what columns are we interested in?
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "policy", "tau", "f", "seed", 
    "N_B", "N_labels", "mean_width", "coverage"
]

# create a log file to store in our "logs" directory
logs_fname = f"baselines_dataset={dataset}_ms={int(sys.argv[2])}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")


'''
Returns: outputs_f1, outputs_f2, where outputs_fi = (N_labels, mean_width_fi, coverage_fi)
'''
# function for running one trial + splitting out the metrics of interest
def trial(M2, PHATS, N_NEW, N_QUESTIONS, N_B, policy, tau, seed, device, counter):
    
    # initialize vectors of (a) thetahat point estimates; (b) widths of CIs with f1(i) = 0.0
    thetahats_unnormalized_f1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    vars_unnormalized_f1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    
    # initialize vectors of (a) thetahat point estimates; (b) widths of CIs with f2(i) = PHATS[i]
    thetahats_unnormalized_f2 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    vars_unnormalized_f2 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    
    # how many labels have we had per new model?
    obs_counts = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    
    # set a seed for reproducibility
    torch.random.manual_seed(seed)
    
    # initialize a ones vector just to be amenable to broadcasting with flipping coins + initialize f1 for readability
    ONES = torch.ones(size=(N_NEW, 1), device=device)
    f1 = torch.tensor(0.0, device=device)
    
    # precompute u(x) scores depending on the policy and other
    U_SCORES = torch.sqrt(PHATS * (1.0 - PHATS)) if policy == "sqrt" else 2.0 * torch.minimum(1.0 - PHATS, PHATS)
    ETA_T = N_B / (N_QUESTIONS * U_SCORES.mean())
    
    # traditional active inference requires looping thru the entire dataset
    for t in tqdm(range(N_QUESTIONS), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):
    
        #### 1. ACTIVE INFERENCE
    
        # simpler world of just uniform sampling ("human")
        if policy == "unif":
    
            # simply uniform
            pi_t_tau = (N_B / N_QUESTIONS) * ONES
    
        # need to do active inference setup.
        elif policy in ["sqrt", "2min"]:
    
            # compute hypothetical remaining budget, adjust according to (Eqn 8 + Algorithm 2, Zrnic & Candes '24)
            n_delta_ts = ((t+1) * N_B / N_QUESTIONS) - obs_counts # (N_NEW, 1)
            pi_t = torch.minimum(ETA_T * U_SCORES[t],  n_delta_ts).clamp(min=0.0, max=1.0) # (N_NEW, 1)
    
            # mixing with uniform -- Section 7.2, Zrnic & Candes '24
            pi_t_tau = ((1.0 - tau) * pi_t) + (tau * (N_B / N_QUESTIONS)) # (N_NEW, 1)
            
        # something went wrong.
        else:
    
            # throw an error.
            raise Exception(f"Policy '{policy}' not supported.")
    
        # are we going to query this data point or not?
        xi_t = torch.bernoulli(pi_t_tau * ONES) # (N_NEW, 1)
            
        # what are predictions for this data point?
        f2 = PHATS[t] # already defined f1 = torch.tensor(0.0, device=device)
    
        # update our means and variances for the variant of f1.
        thetahats_unnormalized_f1 += ((xi_t / pi_t_tau) * (M2[:,t:t+1] - f1)) + f1 # (N_NEW, 1)
        vars_unnormalized_f1 += (xi_t * (1.0 - pi_t_tau) / (pi_t_tau ** 2) ) * ((M2[:,t:t+1] - f1) ** 2) # (N_NEW, 1)
    
        # update our means and variances for the variant of f2.
        thetahats_unnormalized_f2 += ((xi_t / pi_t_tau) * (M2[:,t:t+1] - f2)) + f2 # (N_NEW, 1)
        vars_unnormalized_f2 += (xi_t * (1.0 - pi_t_tau) / (pi_t_tau ** 2) ) * ((M2[:,t:t+1] - f2) ** 2) # (N_NEW, 1)
    
        #### 3. BOOKKEEPING
    
        # update our counters
        obs_counts += xi_t
    
    # get the true means + pre-compute the z-score multiplier
    mus_M2 = M2.mean(dim=1, keepdim=True)
    z_score = norm.ppf(1.0 - (ALPHA / 2))
    
    # compute our estimates + variances for the variant with f1
    thetahats_f1 = thetahats_unnormalized_f1 / N_QUESTIONS
    sigmahats_f1 = torch.sqrt(vars_unnormalized_f1 / N_QUESTIONS)
    sigmahats_bar_f1 = sigmahats_f1 / torch.sqrt(torch.tensor(N_QUESTIONS, device=device))
    lb_f1 = torch.maximum(thetahats_f1 - z_score * sigmahats_bar_f1, torch.tensor(0.0, device=device))
    ub_f1 = torch.minimum(thetahats_f1 + z_score * sigmahats_bar_f1, torch.tensor(1.0, device=device))
    mean_width_f1 = (ub_f1 - lb_f1).mean().item()
    coverage_f1 = ((lb_f1 <= mus_M2) & (mus_M2 <= ub_f1)).mean(dtype=float).item()
    
    # compute our estimates + variances for the variant with f2
    thetahats_f2 = thetahats_unnormalized_f2 / N_QUESTIONS
    sigmahats_f2 = torch.sqrt(vars_unnormalized_f2 / N_QUESTIONS)
    sigmahats_bar_f2 = sigmahats_f2 / torch.sqrt(torch.tensor(N_QUESTIONS, device=device))
    lb_f2 = torch.maximum(thetahats_f2 - z_score * sigmahats_bar_f2, torch.tensor(0.0, device=device))
    ub_f2 = torch.minimum(thetahats_f2 + z_score * sigmahats_bar_f2, torch.tensor(1.0, device=device))
    mean_width_f2 = (ub_f2 - lb_f2).mean().item()
    coverage_f2 = ((lb_f2 <= mus_M2) & (mus_M2 <= ub_f2)).mean(dtype=float).item()
    
    # let's do a budget check
    N_labels = obs_counts.mean().item()
    
    # return our outputs
    outputs_f1, outputs_f2 = [N_labels, mean_width_f1, coverage_f1], [N_labels, mean_width_f2, coverage_f2]
    return outputs_f1, outputs_f2


# a counter to check our progress
counter = 0

# for checkpointing purposes
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index) // 2

# go thru all the datasets and missingness settings
for missingness_setting in MISSINGNESS_SETTINGS:

    # unpack the settings + load in the factor model for this setting
    n_full_obs, mcar_obs_prob = missingness_setting

    # we only need to load in M2
    M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32))
    M2 = M2.to(device)

    # how many new models do we have and how many new qustions do we have?
    N_NEW, N_QUESTIONS = M2.shape

    # load in our partialy-observed M1 matrix + get the means, impute global mean if necessary
    M1 = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
    M1 = torch.tensor(M1.iloc[:,3:].to_numpy().astype(np.float32))
    PHATS = M1.nanmean(dim=0)
    PHATS[PHATS.isnan()] = PHATS.nanmean()

    # go through our budget proportions, beta0, rho, and seeds
    for budget_prop in BUDGET_PROPS:

        # what is our target budget?
        N_B = int(N_QUESTIONS * budget_prop)

        # what is our policy? This will dictate what our taus are
        for policy in POLICIES:

            # uniform sampling policy does not have a tau hyperparameter!
            WORKING_TAUS = [np.nan] if policy == "unif" else TAUS

            # go thru the relevant taus.
            for tau in WORKING_TAUS:
    
                # just go thru all of our seeds
                for seed in range(N_SEEDS):
    
                    # checkpointing
                    if counter >= checkpoint_counter:
    
                        # run our trial - outputs (N_labels, mean_width, coverage) for f1-driven and f2-driven algorithms.
                        outputs_f1, outputs_f2 = trial(M2, PHATS, N_NEW, N_QUESTIONS, N_B, policy, tau, seed, device, counter)

                        # create our rows to write our .csv
                        row_f1 = [dataset, n_full_obs, mcar_obs_prob, budget_prop, policy, tau, "zero", seed, N_B] + outputs_f1
                        row_f2 = [dataset, n_full_obs, mcar_obs_prob, budget_prop, policy, tau, "mean", seed, N_B] + outputs_f2
    
                        # write to .csv - f1-driven (i.e., zero as our ML estimator)
                        with open(f"logs/final/{logs_fname}", "a") as file:
                            file.write(",".join([str(entry) for entry in row_f1]))
                            file.write("\n")

                        # write to .csv - f2-driven (i.e., zero as our ML estimator)
                        with open(f"logs/final/{logs_fname}", "a") as file:
                            file.write(",".join([str(entry) for entry in row_f2]))
                            file.write("\n")
    
                    # increment our counter
                    counter += 1
    
                    # status update
                    if counter % 100 == 0:
                        os.system("clear")
                        print(f"Finished {counter} of {NUM_SETTINGS} settings.")
