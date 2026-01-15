import numpy as np
import pandas as pd
import os, shutil, gc
from tqdm.notebook import tqdm

# huggingface utilities
import huggingface_hub
from huggingface_hub import login, HfApi
login(token="YOUR TOKEN HERE")
from datasets import load_dataset

# start our api
api = HfApi()

# number of questons in MMLU-Pro
MMLU_PRO_SIZE = 12032

# get information on which datasets might contain what we want (i.e., "-details")
all_datasets = api.list_datasets(
    author="open-llm-leaderboard", sort="created_at", direction=1)
qa_datasets = [ds for ds in all_datasets if ds.id.endswith("-details")]

# how many models are there total?
TOTAL_MODELS = len(qa_datasets) # should have 4500 models (theoretically)

# create a .csv if it doesn't exist to store our models' scores on MMLU-Pro
if "mmlu_pro_scores.csv" not in os.listdir("raw"):
    columns = ["model", "created_date", "sha"] + list(np.arange(MMLU_PRO_SIZE).astype(str))
    with open("raw/mmlu_pro_scores.csv", "wt") as file:
        file.write(",".join(columns) + "\n")
        
# let's do some checkpointing - only work on models that we don't have in our .csv
df = pd.read_csv("raw/mmlu_pro_scores.csv").sort_values(by="created_date").reset_index(drop=True).dropna()
finished_models = list(df.model.values)
qa_datasets = [
    ds for ds in tqdm(qa_datasets)
    if ds.id.split("open-llm-leaderboard/")[1].split("-details")[0] not in finished_models]
del df; gc.collect()
print(f"From the previous checkpoint, we have {len(qa_datasets)} models to process.")

# list of problematic (model, dataset) pairs + counter
errors, error_counter, counter = [], 0, len(finished_models)

# go through each of our candidate qa-datasets (each corresponding to a model)
for i, ds in enumerate(qa_datasets):
    
    # get the name of our model + unique SHA identifier
    model_name = ds.id.split("open-llm-leaderboard/")[1].split("-details")[0]
    sha = ds.sha
    
    # see if this model has an MMLU-Pro setup
    try:
        
        # load the dataset
        data = load_dataset(ds.id, name=f"{model_name}__leaderboard_mmlu_pro", split="latest")
        
        # get the accuracy "acc" parameters + write to our .csv efficiently
        scores = data.to_pandas().sort_values(by="doc_id").acc.values.astype(np.int8)
        scores_string = ",".join(scores.astype(str))
        with open("raw/mmlu_pro_scores.csv", "at") as file:
            file.write(f"{model_name},{str(ds.created_at)},{sha}," + scores_string + "\n")
            
        # clear memory + local storage
        data.cleanup_cache_files()
        shutil.rmtree("YOUR CACHE DIR HERE/huggingface/datasets") # replace with your appropriate directory.
        shutil.rmtree("YOUR CACHE DIR HERE/huggingface/hub") # replace with your appropriate directory.
        del data, scores, scores_string
        gc.collect()
        
    except:
        errors.append(model_name)
        error_counter += 1
    
    # status report 
    counter += 1
    os.system("clear")
    print(f"Finished {model_name}.\n\t- Tried {counter} of {TOTAL_MODELS} Models; Failed Models: {error_counter}.\n\t- Last Failed Model: {errors[-1] if len(errors) > 0 else None}.")
    
# load our scraped .csv, sort by date, drop the nan's
df = pd.read_csv("raw/mmlu_pro_scores.csv")
df = df.sort_values(by="created_date").reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

# what is our total number of models
TOTAL_MODELS = len(df.index)

# get our train/test splits
df.iloc[: (TOTAL_MODELS // 2)].to_csv("processed/mmlu-pro/M1.csv", index=False)
df.iloc[(TOTAL_MODELS // 2) :].to_csv("processed/mmlu-pro/M2.csv", index=False)