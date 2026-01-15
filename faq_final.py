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

# we will load in the best hyperparameter settings for each dataset + missingness-setting + prop_budget
best_settings = pd.read_csv("logs/val/best_settings.csv")

# settings that we will get test results for
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
MISSINGNESS_SETTINGS = [(None, 1.0), (50, 0.1), (200, 0.1),  (800, 0.1), (0, 0.01), (0, 0.001), (0, 0.0001), (0, 0.00001)]
BUDGET_PROPS = np.round(np.linspace(0.0, 0.25, 11)[1:], decimals=3)

# fixed hyperparameters + number of seeds for final test purposes
ALPHA, N_SEEDS = 0.05, 100

# let's split into three sets of seeds based on command-line argument (33, 33, 34)
if int(sys.argv[1]) == 0:
    SEED_LIST = np.arange(33)
elif int(sys.argv[1]) == 1:
    SEED_LIST = np.arange(33, 66)
elif int(sys.argv[1]) == 2:
    SEED_LIST = np.arange(66, 100)
    
    
# which columns are we interested in for our logs?
columns = ["dataset", "n_full_obs", "mcar_obs_prob", "prop_budget", "seed", "mean_width", "coverage"]

# create a log file to store in our "logs" directory
logs_fname = f"faq_sl={int(sys.argv[1])}.csv"
if logs_fname not in os.listdir("logs/final"):
    with open(f"logs/final/{logs_fname}", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# function for running one trial and splitting out the metrics of interest
def trial(
    M2, V, MU0, SIGMA0, 
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

    # set a seed for reproducibility
    torch.random.manual_seed(seed)

    # we will strictly label only N_B questions!
    for s in tqdm(range(N_B), desc=f"{str(counter).zfill(5)}"):

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
        z_Is = torch.gather(M2, dim=1, index=I_s)
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
    mus_M2 = M2.mean(dim=1, keepdim=True)

    # compute the mean widths + coverages of the full variance expression
    ub = torch.maximum(
        thetahats_T + z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(0.0, device=device))
    lb = torch.minimum(
        thetahats_T - z_score * torch.sqrt(v_T_sq_full / N_B), torch.tensor(1.0, device=device))
    mean_width = (ub - lb).mean().item()
    coverage = (((lb <= mus_M2) & (mus_M2 <= ub))).mean(dtype=float).item()
    
    # return our outputs
    outputs = [mean_width, coverage]
    return outputs


# how many settings do we have total?
NUM_SETTINGS = len(DATASETS) * len(MISSINGNESS_SETTINGS) * len(BUDGET_PROPS) * len(SEED_LIST)

# a counter to check our progress
counter = 0

# checkpointing to see how many settings we've already finished.
checkpoint_counter = len(pd.read_csv(f"logs/final/{logs_fname}").index)

# iterate through our settings
for dataset in DATASETS:
    
    # we only need to load in M2
    M2 = pd.read_csv(f"data/processed/{dataset}/M2.csv")
    M2 = torch.tensor(M2.iloc[:,3:].to_numpy().astype(np.float32))
    M2 = M2.to(device)
    
    # how many new models do we have and how many new questions do we have?
    N_NEW, N_QUESTIONS = M2.shape
    
    # go thru all of our missingness settings
    for missingness_setting in MISSINGNESS_SETTINGS:
        
        # unpack the settings
        n_full_obs, mcar_obs_prob = missingness_setting
        
        # load our corresponding factor models
        U = torch.load(
            f"factor_models/final/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
        V = torch.load(
            f"factor_models/final/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt").to(device)
        D = U.shape[1]
        
        # initialize starting mean and covariance for our new model factors U_i's
        MU0, SIGMA0 = U.mean(axis=0), torch.cov(U.T)
        
        # go through our budget proportions, beta0, rho, and seeds
        for budget_prop in BUDGET_PROPS:
            
            # what is our budget?
            N_B = int(N_QUESTIONS * budget_prop)
            
            # what is the best setting for this (dataset, n_full_obs, mcar_obs_prob, budget_prop)?
            if mcar_obs_prob != 1.0:
                beta0, rho, gamma, tau = best_settings.query(
                    f"dataset == '{dataset}'" + \
                    f" and n_full_obs == {n_full_obs} and mcar_obs_prob == {mcar_obs_prob}" + \
                    f" and prop_budget == {budget_prop}")\
                [["beta0", "rho", "gamma", "tau"]].values.flatten()
            
            # in case where M1 is fully-observed, need to do a special query to avoid NaN's.
            else:
                beta0, rho, gamma, tau = best_settings.query(
                    f"dataset == '{dataset}'" + \
                    f" and mcar_obs_prob == {mcar_obs_prob}" + \
                    f" and prop_budget == {budget_prop}")\
                [["beta0", "rho", "gamma", "tau"]].values.flatten()
            
            # go thru our 33 or 34 seeds
            for seed in SEED_LIST:
                
                # checkpointing
                if counter >= checkpoint_counter:
                    
                    # run our trial
                    outputs = trial(
                        M2, V, MU0, SIGMA0, 
                        N_NEW, N_QUESTIONS, N_B, 
                        beta0, rho, gamma, tau, seed, device, counter)

                    # create our row to write to our .csv
                    row = [dataset, n_full_obs, mcar_obs_prob, budget_prop, seed] + outputs
                    
                    # write to this .csv manually to avoid pandas overhead.
                    with open(f"logs/final/{logs_fname}", "a") as file:
                        file.write(",".join([str(entry) for entry in row]))
                        file.write("\n")
                
                # increment our counter
                counter += 1
                
                # status update
                if counter % 100 == 0:
                    os.system("clear")
                    print(f"Finished {counter} of {NUM_SETTINGS} settings.")
        
        # clean-house
        del U, V, D, MU0, SIGMA0; gc.collect()
    
    # clean-house one more time
    del M2; gc.collect()