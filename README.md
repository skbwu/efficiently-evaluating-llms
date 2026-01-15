# Efficiently Evaluating LLM Performance with 5x Fewer Queries
This repository accompanies the paper "Efficiently Evaluating LLM Performance with 5x Fewer Queries" submitted to ICML 2026.

**A note on data:** the scraped historical data $M_1$ (denoted as $M$ in the paper) and simulated new models $M_2$ can be found in the `data/processed` folder. This folder contains `mmlu-pro.zip` and `bbh+gpqa+ifeval+math+musr.zip`, containing our two aggregated benchmark suites under varying historical data-missingness levels. Please unzip these two compressed files before running our source code. The descriptor `nfobs` refers to the number of fully-observed rows (with `nfobs=None` referring to all rows fully-observed), while `p` refers to the MCAR elementwise probability of observing the remaining entries. We hope future researchers will benefit from these compiled finite-bank datasets for their own exciting ideas!

**A note on compute:** all of the experiments in this paper were run with one NVIDIA H100 GPU with 80 GB of RAM, with any single parallelized run completing in well under one minute.

**A note on reproducibility:** all main text and appendix figures in our paper can be reproduced exactly using the code provided in this repository. For transparency, we also include all log files required to reproduce results in our `logs` directory, including the following:
- The best factor model hyperparameters per dataset and missingness-level in `factor_model_selected_settings.csv`, with logs of all cross-validated settings in `factor_model_cv_logs.csv`.
- The best Factorized Active Querying (FAQ) hyperparameters as selected through our historical train/val splits in `val/best_settings.csv`, with logs of all hyperparameter variants during our train/val splits in `val/faq_val_logs.csv`.
- FAQ performances (using the hyperparameters selected in the historical train/val splits) on the test models ($M_2$) in `final/faq_final_logs.csv`, baseline test performances (all possible hyperparameter settings) in `final/baseline_logs.csv`, and traditional active inference ablation test performances (all possible hyperparameter settings) in `final/active_inference_ablation.csv`.
- Audit logs of coverage over model release date and model accuracy for Figure 4 (main text) and Figure 9 (appendix) in `coverage_analysis/coverage_analysis_dataset=bbh+gpqa+ifeval+math+musr.csv` and `coverage_analysis/coverage_analysis_dataset=mmlu-pro.csv`.

**Workflow for reproducing all main text and appendix results and figures:**
1. Raw historical data $M_1$ and simulated new models $M_2$ were scraped from HuggingFace's [Open LLM Leaderboard](https://huggingface.co/open-llm-leaderboard) using `data/aggregator_yes_mmlu_pro.py` and `data/aggregator_no_mmlu_pro.py` for MMLU-Pro and BBH+GPQA+IFEval+MATH+MuSR, respectively. Please input your own HuggingFace API key and change the HuggingFace cache directories as needed.
2. Missingness was simulated on the historical data via `missingness_data_generator.py`.
3. Factor model hyperparameter cross-validation was performed via `factor_models_cv.py` (and analyzed in `analyzing_factor_models.py`), with fitted factors for FAQ hyperparameter tuning computed in `factor_models_val.py` and final fitted factors for FAQ deployed on test models computed in `factor_models_final.py`.
4. 

*Note that some scripts described above may include command-line arguments that group the list of seed and parameter combinations into smaller sublists (i.e., embarassingly-parallel computation).

