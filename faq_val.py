import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import sigmoid
import sys, os, gc
from tqdm.autonotebook import tqdm
from scipy.stats import norm

# let's set a device too
device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Hyperparameters to tune over / simulation settings to consider:
1. Datasets: MMLU-Pro and Non-MMLU-Pro
2. Missingness settings: (n_full_obs \in {50, 200, 800} x mcar_obs_prob=0.1) + 100% fully-obs + (n_full_obs=0 x {0.01, 0.001, 0.0001, 0.00001})
3. Budget Proportion Values: 10 evenly-spaced values between 0.025 and 0.25
4. Maximum Tempering: 0.25, 0.5, 0.75, or 1.0
5. Rho (proportion of budget spent on learning U): 0.0, 0.05, 0.25, 0.5, 0.75
6. Gamma (proportion of budget before maximum tempering): 0.0, 0.05, 0.25, 0.5, 0.75
'''
# store all of these simulation settings + hyperparameters as lists
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"] # in cmd-line
MISSINGNESS_SETTINGS = [(50, 0.1), (200, 0.1), (800, 0.1), (None, 1.0), (0, 0.01), (0, 0.001), (0, 0.0001), (0, 0.00001)] # 8 such settings
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)
BETA0_VALS = [0.25, 0.5, 0.75, 1.0]
RHO_VALS = [0.0, 0.05, 0.25, 0.5, 0.75]
GAMMA_VALS = [0.0, 0.05, 0.25, 0.5, 0.75] # in cmd-line
TAU_VALS = [0.05, 0.25, 0.5, 0.75]

# fixed hyperparameters + number of seeds for validation purposes
ALPHA, N_SEEDS = 0.05, 5

# which arguments are we running in this script?
dataset = DATASETS[int(sys.argv[1])]
gamma = GAMMA_VALS[int(sys.argv[2])]
MISSINGNESS_SETTINGS = MISSINGNESS_SETTINGS[:4] if int(sys.argv[3]) == 0 else MISSINGNESS_SETTINGS[4:]

# what columns are we interested in?
columns = [
    "dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", 
    "beta0", "rho", "gamma", "tau", "seed", 
    "mean_width", "coverage_partial", "coverage_full", # metrics with the full variance expression
    "mean_width_simp", "coverage_partial_simp", "coverage_full_simp", # metrics w/o the second var term.
    "num_resampled" # across all N_NEW * N_QUESTIONS, how many model-question pairs got sampled > 1 time?
]

# create a log file to store in our "logs" directory
logs_fname = f"dataset={dataset}_gamma={gamma}_ms={int(sys.argv[3])}.csv"
if logs_fname not in os.listdir("logs/val"):
    with open(f"logs/val/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# function for running one trial and splitting out the metrics of interest
def trial(
    M1_val, M1_full_val, V, MU0, SIGMA0, 
    N_NEW, N_QUESTIONS, N_B, 
    beta0, rho, gamma, tau, seed, device, counter):

    # initialize our factor model starting parameters + indicators of which questions are unobserved
    Uhats = torch.tile(MU0, dims=(N_NEW, 1)).to(device) # (N_NEW, D)
    Sigmahats = torch.tile(SIGMA0, dims=(N_NEW, 1, 1)).to(device) # (N_NEW, D, D)

    # initialize UNNORMALIZED means
    thetahats = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)

    # \sum_t (z_It - \phat_It^t)^2 / q_t(I_t)^2 
    varhats_main = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)

    # \sum_s (z_Is / q_s(I_s))
    varhats_inner1 = torch.zeros(size=(N_NEW, 1), dtype=torch.float32, device=device)

    # store separately \sum_{j=1}^Nq \phat_j^{(t)} for each value of t
    varhats_inner2 = torch.zeros(size=(N_NEW, N_B), dtype=torch.float32, device=device)

    # for insights, let's also see how much datapoints get resampled
    obs_counts = torch.zeros(size=(N_NEW, N_QUESTIONS), dtype=torch.float32, device=device)

    # set a seed for reproducibility
    torch.random.manual_seed(seed)

    # we will strictly label only N_B questions!
    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}", disable=not sys.stderr.isatty()):

        #### PART 1: Computing q_s(j) probabilities + sampling

        # a. compute phats implied by our current factor model, phat*(1-phat)
        p_hat_js = sigmoid(Uhats @ V.T).clamp(min=1e-12, max=1.0 - 1e-12) # (N_NEW, N_QUESTIONS)
        p1mp_hat_js = p_hat_js * (1.0 - p_hat_js) # (N_NEW, N_QUESTIONS)
        sqrt_p1mp_hat_js = torch.sqrt(p_hat_js * (1.0 - p_hat_js)) # (N_NEW, N_QUESTIONS)

        # b. get our oracle-inspired score h_o^s(j)
        ho_js = sqrt_p1mp_hat_js / sqrt_p1mp_hat_js.sum(dim=1, keepdim=True) # (N_NEW, N_QUESTIONS)

        # c. to stay safe, symmetrize our Sigmahats
        Sigmahats = (Sigmahats + Sigmahats.mT) / 2.0

        # d. get our active-learning inspired score h_a^s(j) - everything here is (N_NEW, N_QUESTIONS)
        vtSigmav_js = torch.einsum("ni,mij,nj->mn", V, Sigmahats, V) # faster than bmm.
        log_denominator = torch.log1p(p1mp_hat_js * vtSigmav_js)
        sq_term = torch.bmm(Sigmahats, ((p1mp_hat_js @ V) / N_QUESTIONS).unsqueeze(-1)).squeeze(-1) @ V.T
        log_numerator = torch.log(p1mp_hat_js.clamp_min(1e-12)) + 2 * torch.log(sq_term.abs().clamp_min(1e-12))
        log_d_js = log_numerator - log_denominator
        ha_js = torch.softmax(log_d_js, dim=1)

        # d. compute our alpha_s exploration and beta_s tempering governors
        alpha_s = torch.maximum( torch.tensor(0.0),  torch.tensor(1.0 - ( (s+1.0) / (rho * N_B)) )) if rho != 0.0 else 0.0
        beta_s = beta0 * torch.minimum( torch.tensor((s+1.0) / (gamma * N_B)), torch.tensor(1.0)) if gamma != 0.0 else beta0

        # d. combine our scores together + drive down previous observed entries
        hcat_js = ( ((1.0 - alpha_s) * ho_js) + (alpha_s * ha_js) ) ** beta_s

        # e. normalize, mix with tau-governed uniform, and get our actual probabilities
        q_js = ((hcat_js / hcat_js.sum(dim=1, keepdim=True)) * (1.0 - tau)) + (tau / N_QUESTIONS)

        # f. which datapoints are we summoning as our indices?
        I_s = torch.multinomial(input=q_js, num_samples=1)

        #### PART 2: Updating Our FAQ Estimators

        '''
        Remark:
        - Suppose we are in a case where during validation, I can query my LLMs, so my M1 can grow a bit.
        '''
        # a. gather the true labels, predicted probabilities, and labeling probabilities
        z_Is = torch.gather(M1_full_val, dim=1, index=I_s)
        phat_Is = torch.gather(p_hat_js, dim=1, index=I_s)
        q_Is = torch.gather(q_js, dim=1, index=I_s)

        # b. phi_s = [\sum_j \phat_j^{(s)}] + ((z_{I_s} - \phat_{I_s}^{(s)}) / q_s(I_s))
        prob_sums = p_hat_js.sum(dim=1, keepdim=True)
        aipw_s = (z_Is - phat_Is) / q_Is
        phi_s = prob_sums + aipw_s
        thetahats += phi_s

        # c. variance term updates
        varhats_main += (aipw_s ** 2)
        varhats_inner1 += (z_Is / q_Is)
        varhats_inner2[:,s] = prob_sums.flatten()

        #### PART 3: UPDATING FACTOR MODEL

        # covariance matrix updates
        w_s = torch.gather(input=p1mp_hat_js, dim=1, index=I_s) # (N_NEW, 1)
        v_Is = V[I_s.flatten()] # (N_NEW, D)
        Sigma_v_Is = torch.bmm(Sigmahats, v_Is.unsqueeze(-1)) # (N_NEW, D, 1)
        vT_Sigma_v_Is = torch.bmm(v_Is.unsqueeze(-2), Sigma_v_Is).squeeze(-1) # (N_NEW, 1)
        denominator = 1.0 + (w_s * vT_Sigma_v_Is) # (N_NEW, 1)
        numerator = w_s.unsqueeze(dim=-1) * (Sigma_v_Is @ Sigma_v_Is.mT) # (N_NEW, D, D)
        Sigmahats -= (numerator / denominator.unsqueeze(dim=-1))

        # mean vector updates
        Uhats += torch.bmm(Sigmahats, ((z_Is - phat_Is) * v_Is).unsqueeze(dim=-1)).squeeze()

        #### PART 4. UPDATING COUNTERS

        # update our obs. counts per index
        obs_counts.scatter_add_(dim=1, index=I_s, src=torch.ones_like(I_s, dtype=torch.float32))

    # computing coverage + width metrics
    thetahats_T = thetahats / (N_B * N_QUESTIONS)
    v_T_sq_simp = varhats_main / (N_B * (N_QUESTIONS ** 2))

    # the second term that we are subtracting out
    v_T_sq_minus = (((varhats_inner1 / N_B) - varhats_inner2) ** 2).sum(dim=1, keepdim=True)
    v_T_sq_minus /= (N_B * (N_QUESTIONS ** 2))

    # compute the full v_T_sq
    v_T_sq_full = (v_T_sq_simp - v_T_sq_minus).clamp(min=0)

    # get the z_score
    z_score = norm.ppf(1 - (ALPHA / 2))

    # what were the true means?
    mus_val = M1_val.mean(dim=1, keepdim=True)
    mus_full_val = M1_full_val.mean(dim=1, keepdim=True)

    # compute the mean widths + coverages of the full variance expression
    ub_full = torch.maximum(
        thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(0.0, device=device))
    lb_full = torch.minimum(
        thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(1.0, device=device))
    mean_width_full = (ub_full - lb_full).mean().item()
    coverage_full_full = (((lb_full <= mus_full_val) & (mus_full_val <= ub_full))).mean(dtype=float).item()
    coverage_partial_full = (((lb_full <= mus_val) & (mus_val <= ub_full)))[~mus_val.isnan()].mean(dtype=float).item()

    # compute the mean widths + coverages of the simp variance expression
    ub_simp = torch.maximum(
        thetahats_T + z_score * torch.sqrt(v_T_sq_simp / N_B), torch.tensor(0.0, device=device))
    lb_simp = torch.minimum(
        thetahats_T - z_score * torch.sqrt(v_T_sq_simp / N_B), torch.tensor(1.0, device=device))
    mean_width_simp = (ub_simp - lb_simp).mean().item()
    coverage_full_simp = (((lb_simp <= mus_full_val) & (mus_full_val <= ub_simp))).mean(dtype=float).item()
    coverage_partial_simp = (((lb_simp <= mus_val) & (mus_val <= ub_simp)))[~mus_val.isnan()].mean(dtype=float).item()

    # how many model-question pairs were queried more than once?
    num_resampled = (obs_counts > 1).sum(dtype=float).item()
    
    # return our outputs
    outputs = [
        mean_width_full, coverage_partial_full, coverage_full_full, 
        mean_width_simp, coverage_partial_simp, coverage_full_simp, num_resampled]
    return outputs

# a counter to check our progress
counter = 0

# checkpointing to see how many settings we've already finished.
checkpoint_counter = len(pd.read_csv(f"logs/val/{logs_fname}").index)

# what's the total number of settings
NUM_SETTINGS = len(MISSINGNESS_SETTINGS) * len(BUDGET_PROPS) * len(BETA0_VALS)\
* len(RHO_VALS) * len(TAU_VALS) * N_SEEDS

# go thru all of our missingness settings
for missingness_setting in MISSINGNESS_SETTINGS:
    
    # unpack the settings
    n_full_obs, mcar_obs_prob = missingness_setting
    
    # load in our M1, split into train + val portions (0.8 vs. 0.2)
    M1 = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
    M1 = torch.tensor(M1.iloc[:,3:].to_numpy().astype(np.float32))
    M1_train, M1_val = M1[:int(M1.shape[0] * 0.8)], M1[int(M1.shape[0] * 0.8):]
    M1_train, M1_val = M1_train.to(device), M1_val.to(device)
    
    # we'll also load in M1_val (fully-observed) to simulate adding data to M1 during the validation phase.
    M1_full = pd.read_csv(f"data/processed/{dataset}/M1_nfobs=None_p=1.0.csv")
    M1_full = torch.tensor(M1_full.iloc[:,3:].to_numpy().astype(np.float32))
    M1_full_val = M1_full[int(M1_full.shape[0] * 0.8):]
    M1_full_val = M1_full_val.to(device)
    
    # how many new models do we have and how many new qustions do we have?
    N_NEW, N_QUESTIONS = M1_val.shape
    
    # load our corresponding factor models
    if n_full_obs != 0:
        U = torch.load(f"factor_models/val/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
        V = torch.load(f"factor_models/val/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
    D = U.shape[1]
    
    # initialize starting mean and covariance for our new model factors U_i's
    MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)
    
    # go through our budget proportions, beta0, rho, and seeds
    for budget_prop in BUDGET_PROPS:
        
        # what is our budget?
        N_B = int(N_QUESTIONS * budget_prop)
        
        # go thru all these settings
        for beta0 in BETA0_VALS:
            for rho in RHO_VALS:
                for tau in TAU_VALS:
                    for seed in range(N_SEEDS):
                        
                        # checkpointing
                        if counter >= checkpoint_counter:
                        
                            # run our trial
                            outputs = trial(
                                M1_val, M1_full_val, V, MU0, SIGMA0, 
                                N_NEW, N_QUESTIONS, N_B, 
                                beta0, rho, gamma, tau, seed, device, counter)

                            # create our row to write to our .csv
                            row = [
                                dataset, n_full_obs, mcar_obs_prob, budget_prop, 
                                beta0, rho, gamma, tau, seed] + outputs

                            # write to this .csv manually to avoid pandas overhead.
                            with open(f"logs/val/{logs_fname}", "a") as file:
                                file.write(",".join([str(entry) for entry in row]))
                                file.write("\n")
                            
                        # increment our counter
                        counter += 1
                        
                        # status update
                        if counter % 100 == 0:
                            os.system("clear")
                            print(f"Finished {counter} of {NUM_SETTINGS} settings.")
    
    # clean house after finishing this wave of missingness settings
    del M1, M1_train, M1_val, M1_full, M1_full_val, U, V, D, MU0, SIGMA0
    gc.collect()
