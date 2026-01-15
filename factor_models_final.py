import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import torch, gc
from torch.optim import AdamW
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits as BCE

# set our device + number of epochs
device = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 2000

# what are the settings we will be fitting factor models on the FULL M1 training data?
selected_settings = pd.read_csv("logs/factor_model_selected_settings.csv")

# go thru each row of our selected settings
for idx in selected_settings.index:
    
    # unpack the settings + adjust if necessary
    dataset, n_full_obs, mcar_obs_prob, K, lmbda = selected_settings.loc[idx][:5]
    n_full_obs = None if np.isnan(n_full_obs) else int(n_full_obs)
    
    # what are we working on?
    print(f"Working on dataset={dataset}, nfobs={n_full_obs}, mcar_obs_prob={mcar_obs_prob}.")
    
    # load in our data, BUT TRAIN ON THE ENTIRE M1!
    M1 = pd.read_csv(f"data/processed/{dataset}/M1_nfobs={n_full_obs}_p={mcar_obs_prob}.csv")
    M1 = torch.tensor(M1.iloc[:,3:].to_numpy().astype(np.float32))
    M1 = M1.to(device)
    
    # which entries in M1 are actually observed?
    M1_avail_mask = ~M1.isnan()
    
    # get our parameters for M1
    N_M1, TOTAL_QUESTIONS = M1.shape
    
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
    for epoch in tqdm(range(MAX_EPOCHS), desc=f"K={K}, lmbda={lmbda}"):

        # forward pass
        train_loss = BCE((U @ V.T)[M1_avail_mask], M1[M1_avail_mask])

        # backwards pass, update parameters, reset gradient
        train_loss.backward(); optimizer.step(); optimizer.zero_grad()
    
    # disable gradients
    U.requires_grad = False
    V.requires_grad = False

    # save our U and V
    torch.save(U, f"factor_models/final/{dataset}/U_nfobs={n_full_obs}_p={mcar_obs_prob}.pt")
    torch.save(V, f"factor_models/final/{dataset}/V_nfobs={n_full_obs}_p={mcar_obs_prob}.pt")
    
    # clean house
    del M1, M1_avail_mask, U, V, optimizer
    gc.collect()