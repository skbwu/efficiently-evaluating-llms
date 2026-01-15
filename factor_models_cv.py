import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import torch, os, gc
from torch.optim import AdamW
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCE

# to do cross-validation on the factor model dimension + choice of lmbda
from sklearn.model_selection import KFold

# set our device
device = "cuda" if torch.cuda.is_available() else "cpu"

# create a .csv to store our results (just initialize it for now)
columns = [
    "dataset", 
    "n_full_obs", "mcar_obs_prob",
    "K", "lmbda", "split",
    "train_loss", "val_loss", "train_acc", "val_acc",
]
if "factor_model_cv_logs.csv" not in os.listdir("logs"):
    with open("logs/factor_model_cv_logs.csv", "w") as file:
        file.write(",".join([str(col) for col in columns]) + "\n")

# simulation settings to account for
DATASETS = ["mmlu-pro", "bbh+gpqa+ifeval+math+musr"]
MISSINGNESS_SETTINGS = [(50, 0.1), (200, 0.1), (800, 0.1), (0, 0.01), (0, 0.001), (0.0001), (0.00001)]

# hyperparameters to cv over + number of cv_splits
Ks, lmbdas = [8, 16, 32, 64], [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
N_SPLITS = 5

# total number of settings (not including cv splits)
N_SIM_SETTINGS = len(DATASETS) * (MISSINGNESS_SETTINGS + 1)
N_SETTINGS = N_SIM_SETTINGS * len(Ks) * len(lmbdas) * N_SPLITS

# maximum number of epochs
MAX_EPOCHS = 2000

'''
Runs all settings for one combination of dataset x n_full_obs x mcar_obs_prob and returns a dataframe.
'''
# so far, all of this just requires (n_full_obs, mcar_obs_prob)
def assemble_sim_df(dataset, n_full_obs, mcar_obs_prob, random_state):
    
    # create a dataframe to store this portion of results
    sim_df = pd.DataFrame(columns=columns)
    
    # what are we working on?
    print(f"Working on dataset={dataset}, nfobs={n_full_obs}, mcar_obs_prob={mcar_obs_prob}.")
    
    # load in our data, but keep 20% for validation purposes
    M1 = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
    M1 = torch.tensor(M1.iloc[:,3:].to_numpy().astype(np.float32))
    M1 = M1[:int(M1.shape[0] * 0.8)]; M1 = M1.to(device)

    # which entries in M1 are actually observed?
    M1_avail_mask = ~M1.isnan()

    # get our parameters for M1
    N_M1, TOTAL_QUESTIONS = M1.shape

    # split our data into 5-fold cross-validation splits
    splits = list(KFold(n_splits=5, shuffle=True, random_state=random_state).split(M1.flatten()))

    # go thru each of our cv split
    for split_idx, split in enumerate(splits):

        # get our train and validation indices + construct masks
        train_idxs, val_idxs = split
        train_mask = np.zeros(shape=N_M1 * TOTAL_QUESTIONS, dtype=bool); train_mask[train_idxs] = True
        train_mask = torch.tensor(train_mask.reshape(N_M1, TOTAL_QUESTIONS), device=device)
        val_mask = ~train_mask

        # make sure our masks account for the missingness in the data
        train_mask = train_mask & M1_avail_mask
        val_mask = val_mask & M1_avail_mask

        # how many questions in our training and validation splits?
        N_train, N_val = train_mask.sum(dtype=float), val_mask.sum(dtype=float)

        # grid-search over our hyperparameters
        for K in Ks:
            for lmbda in lmbdas:

                # set a seed for reproducibility
                torch.random.manual_seed(858)

                # initialize our tensors
                U = nn.Parameter(torch.randn(size=(N_M1, K), requires_grad=True, device=device))
                V = nn.Parameter(torch.randn(size=(TOTAL_QUESTIONS, K), requires_grad=True, device=device))

                # instantiate our optimizer
                optimizer = AdamW(
                    lr=1e-2, params=[
                        {"params" : [U, V], "weight_decay" : lmbda}])

                # let's do our training
                for epoch in tqdm(range(MAX_EPOCHS), desc=f"split={split_idx}, K={K}, lmbda={lmbda}"):

                    # forward pass
                    train_loss = BCE((U @ V.T)[train_mask], M1[train_mask])

                    # backwards pass, update parameters, reset gradient
                    train_loss.backward(); optimizer.step(); optimizer.zero_grad()

                # compute metrics at the very end
                with torch.no_grad():

                    # get our yhat on overall M1
                    UVT = U @ V.T
                    yhat = nn.functional.sigmoid(UVT) > 0.5

                    # compute our metrics one final time
                    train_acc = (((yhat == M1) * train_mask).sum() / N_train).item()
                    train_loss = (BCE(UVT[train_mask], M1[train_mask])).item()
                    val_acc = (((yhat == M1) * val_mask).sum() / N_val).item()
                    val_loss = (BCE(UVT[val_mask], M1[val_mask])).item()

                # record our results to our simulation dataframe
                sim_df.loc[len(sim_df.index)] = [
                    dataset, n_full_obs, mcar_obs_prob, K, lmbda, split_idx,
                    train_loss, val_loss, train_acc, val_acc]

                # clean house with memory + wipe
                del U, V, optimizer
                gc.collect()

        # clean house one more time
        del train_idxs, val_idxs, train_mask, val_mask, N_train, N_val
        gc.collect()
        
    # at the end, just clear console + return sim_df
    os.system("clear")
    return sim_df

# go thru all of our datasets
for dataset in DATASETS:
    
    # fill in dummy variables for now on n_full_obs and mcar_obs_prob (fully-observed version!)
    n_full_obs, mcar_obs_prob = None, 1.0

    # let's do some checkpointing
    sim_df = pd.read_csv("logs/factor_model_cv_logs.csv")
    if len(sim_df.query(f"dataset == '{dataset}' and mcar_obs_prob == 1.0").index) == 0:

        # get our dataframe of results for this (dataset, n_full_obs, mcar_obs_prob) + add to our master
        sim_df = assemble_sim_df(dataset, n_full_obs, mcar_obs_prob, 858)
        sim_df.to_csv("logs/factor_model_cv_logs.csv", mode="a", index=False, header=False)
    
    # go thru our missing-data settings
    for missingness_setting in MISSINGNESS_SETTINGS:

        # get our n_full_obs and mcar_obs_prob
        n_full_obs, mcar_obs_prob = missingness_setting

        # let's do some checkpointing
        sim_df = pd.read_csv("logs/factor_model_cv_logs.csv")
        if len(sim_df.query(f"dataset == '{dataset}' and n_full_obs == {n_full_obs} and mcar_obs_prob == {mcar_obs_prob}").index) == 0:
        
            # get our dataframe of results for this (dataset, n_full_obs, mcar_obs_prob) + add to our master
            sim_df = assemble_sim_df(dataset, n_full_obs, mcar_obs_prob, 858)
            sim_df.to_csv("logs/factor_model_cv_logs.csv", mode="a", index=False, header=False)