import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, shutil, gc
from tqdm.autonotebook import tqdm
import pickle
import itertools
import time

# huggingface utilities
import huggingface_hub
from huggingface_hub import login, HfApi
login(token="YOUR TOKEN HERE")
from datasets import load_dataset

# import the dataset details
from data_helpers import DATASETS_AND_SIZES

# start our api
api = HfApi()

# get information on which datasets might contain what we want (i.e., "-details")
all_datasets = api.list_datasets(
    author="open-llm-leaderboard", sort="created_at", direction=1)
qa_datasets = [ds for ds in all_datasets if ds.id.endswith("-details")]

# how many models are there total? How many total questions are we working with?
TOTAL_MODELS = len(qa_datasets)
N_QUESTIONS = np.sum(list(DATASETS_AND_SIZES.values())) # 9574
N_DATASETS = len(DATASETS_AND_SIZES) # 38

# get the columns corresponding to each of our datasets + questions: e.g., "ifeval_11"
question_columns = list(
    itertools.chain.from_iterable([
        [f"{dataset}_{i}" for i in range(DATASETS_AND_SIZES[dataset])] 
         for dataset in DATASETS_AND_SIZES.keys()
        ])
)

# create a .csv if it doesn't exist to store our models' scores on not-MMLU-Pro
if "non_mmlu_pro_scores.csv" not in os.listdir("raw"):
    columns = ["model", "created_date", "sha"] + question_columns
    with open("raw/non_mmlu_pro_scores.csv", "wt") as file:
        file.write(",".join(columns) + "\n")
        
# let's do some checkpointing - only work on models that we don't have in our .csv
df = pd.read_csv("raw/non_mmlu_pro_scores.csv").sort_values(by="created_date").reset_index(drop=True).dropna()
finished_models = list(df.model.values)
qa_datasets = [
    ds for ds in tqdm(qa_datasets)
    if ds.id.split("open-llm-leaderboard/")[1].split("-details")[0] not in finished_models]
del df; gc.collect()
print(f"From the previous checkpoint, we have {len(qa_datasets)} models to process.")

# list of problematic (model, dataset) pairs + counter
errors, error_counter, counter = [], 0, len(finished_models)

# go through each of our candidate qa-datasets (each corresponding to a MODEL)
for i, model in enumerate(qa_datasets):
    
    # start our timer
    start = time.time()
    
    # make the .cache directories to avoid memory issues from non-existent directory purging
    if not os.path.isdir("/Users/skylerwu/.cache/huggingface/datasets"):
        os.makedirs("/Users/skylerwu/.cache/huggingface/datasets", exist_ok=True)
    if not os.path.isdir("/Users/skylerwu/.cache/huggingface/hub"):
        os.makedirs("/Users/skylerwu/.cache/huggingface/hub", exist_ok=True)
    
    # get the name of our model + unique SHA identifier
    model_name = model.id.split("open-llm-leaderboard/")[1].split("-details")[0]
    sha = model.sha
    
    # instantiate an overall "scores" list for all the datasets
    scores = []
    
    # let's go thru each dataset
    for dataset in DATASETS_AND_SIZES.keys():
        
        # see if we can load this dataset or not
        try:
            
            # try loading in the dataset + converting into an appropriate form
            data = load_dataset(model.id, name=f"{model_name}__leaderboard_{dataset}", split="latest")
            data_vals = data.to_pandas().sort_values(by="doc_id")
            
            '''
            Accuracy column names by dataset:
            1. Anything starting with "bbh" - "acc_norm" (1 for correct, 0 for wrong).
            2. Anything starting with "gpqa" - "acc_norm" (1 for correct, 0 for wrong).
            3. For "ifeval" only - "prompt_level_loose_acc" (true for correct, false for wrong).
            4. Anything starting with "math" - "exact_match" (1 for correct, 0 for wrong).
            '''
            # we need to extract the relevant accuracy column
            if dataset.startswith("bbh"):
                vals = data_vals.acc_norm.values.astype(np.int8)
            elif dataset.startswith("gpqa"):
                vals = data_vals.acc_norm.values.astype(np.int8)
            elif dataset.startswith("ifeval"):
                vals = data_vals.prompt_level_loose_acc.values.astype(np.int8)
            elif dataset.startswith("math"):
                vals = data_vals.exact_match.values.astype(np.int8)
            elif dataset.startswith("musr"):
                vals = data_vals.acc_norm.values.astype(np.int8)
            else:
                raise Exception(f"Dataset {dataset} is not supported for {model_name}.")
            
            # add these vals to our "scores" list
            vals = list(vals)
            scores.extend(vals)
            
            # clean house
            data.cleanup_cache_files()
            del data, data_vals, vals
            gc.collect()
        
        except:
            
            # append our lists + counters
            errors.append((model_name, dataset))
            error_counter += 1
            
            # append "scores" with equivalent nan's
            scores.extend([np.nan] * DATASETS_AND_SIZES[dataset])
    
    # process the scores + write to our file
    scores_string = ",".join(np.array(scores).astype(str))
    with open("raw/non_mmlu_pro_scores.csv", "at") as file:
        file.write(f"{model_name},{str(model.created_at)},{sha}," + scores_string + "\n")

    # clear memory + local storage
    shutil.rmtree("YOUR CACHE DIR HERE/huggingface/datasets") # replace with your appropriate cache directory
    shutil.rmtree("YOUR CACHE DIR HERE/huggingface/hub") # replace with your appropriate cache directory
    del scores, scores_string
    gc.collect()        
    
    # stop our timer
    end = time.time()
    time_elapsed = end - start
            
    # after we've gone thru all the datasets for this model, status report
    counter += 1
    os.system("clear")
    print(f"Finished {model_name}.")
    print(f"\t- Tried {counter} of {TOTAL_MODELS} Models.")
    print(f"\t- Last Failed (Model, Dataset): {errors[-1] if len(errors) > 0 else None}.")
    print(f"\t- Number of Failed (Model, Dataset) Pairs: {error_counter} of {TOTAL_MODELS * N_DATASETS}.")
    print(f"\t- Overall Failure Rate: {error_counter * 100 / (TOTAL_MODELS * N_DATASETS):.3f}%.")
    print(f"\t- Finished in {time_elapsed:.3f} seconds.")
    
# load our scraped .csv, sort by date, drop the nan's
df = pd.read_csv("raw/non_mmlu_pro_scores.csv")
df = df.sort_values(by="created_date").reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

# what is our total number of models (4428)
TOTAL_MODELS = len(df.index)

# get our train/test splits
df.iloc[: (TOTAL_MODELS // 2)].to_csv("processed/bbh+gpqa+ifeval+math+musr/M1.csv", index=False)
df.iloc[(TOTAL_MODELS // 2) :].to_csv("processed/bbh+gpqa+ifeval+math+musr/M2.csv", index=False)