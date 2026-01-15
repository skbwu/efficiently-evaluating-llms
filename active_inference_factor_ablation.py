import numpy as np
import pandas as pd
import torch
from torch.nn.functional import sigmoid
import sys, os, gc
from scipy.stats import norm
from tqdm.autonotebook import tqdm

# let's set a device too
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Active Inference + Factor Model Ablation (Zrnic & Candes '24)

Settings to iterate over:
1. Datasets: MMLU-Pro and Non-MMLU-Pro
2. Missingness settings: (n_full_obs=0 x mcar_obs_prob in {0.01, 0.001, 0.0001, 0.00001}) + 100% fully-obs.
3. Budget Proportion Values: 10 evenly-spaced values between 0.025 and 0.25
'''
# datasets + missingness settings, also budget_props
dataset = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"][int(sys.argv[1])]
MISSINGNESS_SETTINGS = [(None, 1.0), (0, 0.00001), (0, 0.0001), (0, 0.001), (0, 0.01)]
BUDGET_PROPS = np.linspace(0.0, 0.25, 11)[1:]

# Zrnic & Candes, Section 7.2: Practical Sampling Rules
TAUS = [0.05, 0.25, 0.5, 0.75]

# fixed hyperparameters
ALPHA, N_SEEDS = 0.05, 100

# how many settings are we running total for this script?
NUM_SETTINGS = len(MISSINGNESS_SETTINGS) * len(BUDGET_PROPS) * len(TAUS) * N_SEEDS

# what columns are we interested in?
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "tau", "seed", 
    "N_B", "N_labels", "mean_width", "coverage"
]

# create a log file to store in our "logs" directory
logs_fname = f"active_inference_ablation_dataset={dataset}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

        
# function for running one trial + splitting out the metrics of interest
def trial(M2, V, MU0, SIGMA0, N_NEW, N_QUESTIONS, N_B, tau, seed, device, counter):

    # initialize vectors of (a) thetahat point estimates; (b) widths of CIs
    thetahats_unnormalized = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)
    vars_unnormalized = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)

    # initialize our factor model starting parameters + counts of how many labels we've used up
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device) # (N_NEW, D)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device) # (N_NEW, D, D)
    obs_counts = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)

    # set a seed for reproducibility
    torch.random.manual_seed(seed)

    # traditional active inference requires looping thru the entire dataset
    for t in tqdm(range(N_QUESTIONS), desc=f"{str(counter).zfill(5)}"):

        #### 1. ACTIVE INFERENCE

        # current estimates of phat_j
        phats_j = sigmoid(Uhats @ V.T)

        # u_t(x) = 2*min(p(x), 1-p(x)) (Eqn 3, Zrnic & Candes '24) + empirical expectation
        u_ts = 2.0 * torch.minimum(phats_j, 1.0 - phats_j) # (N_NEW, N_QUESTIONS)
        Eu_ts = u_ts.mean(dim=1, keepdim=True)

        # what are our label probabilities for this current z_t? (Eqn 8 + Algorithm 2, Zrnic & Candes '24)
        eta_ts = N_B / (N_QUESTIONS * Eu_ts) # (N_NEW, 1)
        n_delta_ts = ((t+1) * N_B / N_QUESTIONS) - obs_counts # (N_NEW, 1)
        pi_t = torch.minimum(eta_ts * u_ts[:,t:t+1], n_delta_ts).clamp(min=0.0, max=1.0) # (N_NEW, 1)

        # mixing with uniform -- Section 7.2, Zrnic & Candes '24
        pi_t_tau = ((1.0 - tau) * pi_t) + (tau * (N_B / N_QUESTIONS)) # (N_NEW, 1)

        # are we going to query this data point or not?
        xi_t = torch.bernoulli(pi_t_tau) # (N_NEW, 1)

        # predictions for this datapoint
        f = phats_j[:,t:t+1]

        # update our means and variances
        thetahats_unnormalized += ((xi_t / pi_t_tau) * (M2[:,t:t+1] - f)) + f # (N_NEW, 1)
        vars_unnormalized += (xi_t * (1.0 - pi_t_tau) / (pi_t_tau ** 2) ) * ((M2[:,t:t+1] - f) ** 2) # (N_NEW, 1)

        #### 3. BOOKKEEPING

        # update our counters
        obs_counts += xi_t

        #### 2. FACTOR MODEL UPDATES

        # convert xi_t to bool
        xi_t = xi_t.to(bool).squeeze() # (N_NEW,)

        # compute our w = p(1-p)
        w_t = f[xi_t] * (1.0 - f[xi_t]) # (labeled-at-t, 1)

        # compute v_j v_j^T
        v_t = V[t].unsqueeze(-1) # (D, 1)

        # compute Sigmahat @ v_t (labeled entries only!)
        Sv_t = Sigmahats[xi_t] @ v_t # (labeled-at-t, D, 1)

        # compute the numerator + denominator + covariance update (Eqn 3, our manuscript)
        numerator = (Sv_t @ Sv_t.mT) * w_t.unsqueeze(dim=-1) # (labeled-at-t, D, D)
        denominator = 1.0 + (w_t.unsqueeze(dim=-1) * (v_t.T @ Sigmahats[xi_t] @ v_t)) # (labeled-at-t, 1, 1)
        Sigmahats[xi_t] -= (numerator / denominator)

        # compute the uhat update (Eqn 4, our manuscript)
        Uhats[xi_t] += ((Sigmahats[xi_t] @ v_t) * (M2[xi_t][:,t:t+1] - f[xi_t]).unsqueeze(dim=-1)).squeeze()

    # get the true means
    mus_M2 = M2.mean(dim=1, keepdim=True)

    # compute our estimates + variances
    thetahats = thetahats_unnormalized / N_QUESTIONS
    sigmahats = torch.sqrt(vars_unnormalized / N_QUESTIONS)
    sigmahats_bar = sigmahats / torch.sqrt(torch.tensor(N_QUESTIONS, device=device))

    # get our intervals + check our widths and coverage
    z_score = norm.ppf(1 - (ALPHA / 2))
    lb = torch.maximum(thetahats - z_score * sigmahats_bar, torch.tensor(0.0, device=device))
    ub = torch.minimum(thetahats + z_score * sigmahats_bar, torch.tensor(1.0, device=device))
    mean_width = (ub - lb).mean().item()
    coverage = ((lb <= mus_M2) & (mus_M2 <= ub)).mean(dtype=float).item()

    # let's do a budget check
    N_labels = obs_counts.mean().item()
    
    # return our outputs
    outputs = [N_labels, mean_width, coverage]
    return outputs


# a counter to check our progress
counter = 0

# for checkpointing purposes
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index)

# go thru all the datasets and missingness settings
for missingness_setting in MISSINGNESS_SETTINGS:

    # unpack the settings + load in the factor model for this setting
    n_full_obs, mcar_obs_prob = missingness_setting
    U = torch.load(
        f"factor_models/final/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt", map_location=device)
    V = torch.load(
        f"factor_models/final/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt", map_location=device)
    D = U.shape[1]

    # we only need to load in M2
    M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32))
    M2 = M2.to(device)

    # how many new models do we have and how many new qustions do we have?
    N_NEW, N_QUESTIONS = M2.shape

    # initialize starting mean and covariance for our new model factors U_i's
    MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)

    # go through our budget proportions, beta0, rho, and seeds
    for budget_prop in BUDGET_PROPS:
        for tau in TAUS:

            # what is our budget?
            N_B = int(N_QUESTIONS * budget_prop)

            # just go thru all of our seeds
            for seed in range(N_SEEDS):

                # checkpointing
                if counter >= checkpoint_counter:

                    # run our trial
                    outputs = trial(
                        M2, V, MU0, SIGMA0, N_NEW, N_QUESTIONS, N_B, tau, seed, device, counter)

                    # create our row to write our .csv
                    row = [dataset, n_full_obs, mcar_obs_prob, budget_prop, tau, seed, N_B] + outputs

                    # write to .csv
                    with open(f"logs/final/{logs_fname}", "a") as file:
                        file.write(",".join([str(entry) for entry in row]))
                        file.write("\n")

                # increment our counter
                counter += 1

                # status update
                if counter % 100 == 0:
                    os.system("clear")
                    print(f"Finished {counter} of {NUM_SETTINGS} settings.")

