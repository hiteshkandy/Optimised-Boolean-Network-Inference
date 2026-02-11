BioScore refactor scaffold

Usage:
  1) Place this folder in the same working directory where analyse_database13.py and canalizing_function_toolbox_v16.py are importable.
  2) Run:
       python run_pipeline.py
  3) Outputs:
       all_scores_per_network.csv
       corr_functional_scores.csv
       corr_network_scores_with_motifs.csv

Tests:
  pytest -q
