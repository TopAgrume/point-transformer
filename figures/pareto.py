"""
pareto.py
Custom implementation.

Author: Alexandre Devaux Rivière
Project: NPM3D
Date: 20/03/2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({"figure.autolayout": True})

def load_data(csv_filepath):
    return pd.read_csv(csv_filepath, names=["run", "tag", "step", "value", "wall_time"], header=0)

def plot_pareto_efficiency(df, study_runs, labels_dict, title, filename):
    """scatter plot -> accuracy vs training time"""
    results = []
    for run in study_runs:
        run_data = df[df['run'] == run]
        if run_data.empty: continue

        max_oa = run_data[run_data['tag'] == 'allAcc_val']['value'].max()

        min_time = run_data['wall_time'].min()
        max_time = run_data['wall_time'].max()
        time_hours = (max_time - min_time) / 3600

        results.append({
            "Configuration": labels_dict[run],
            "Max_OA": max_oa,
            "Training_time": time_hours
        })

    res_df = pd.DataFrame(results)

    plt.figure(figsize=(7, 5))
    ax = sns.scatterplot(
        data=res_df, x="Training_time", y="Max_OA",
        hue="Configuration", s=150, palette="deep"
    )

    # text next to nodes
    for i in range(res_df.shape[0]):
        plt.text(res_df["Training_time"].iloc[i] + 0.05,
                 res_df["Max_OA"].iloc[i] + 0.0002,
                 res_df["Configuration"].iloc[i],
                 fontsize=10)

    plt.title(title)
    plt.xlabel("Total training time (h)")
    plt.ylabel("Max overall accuracy (test)")
    plt.grid(True, linestyle='--', alpha=0.7)
    ax.get_legend().remove() # no need
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()

df = load_data("figures/csv/tensorboard_all_runs.csv")
k_runs = ['exp_k_4', 'exp_k_8', 'exp_k_16', 'exp_k_24']
k_labels = {'exp_k_4': 'k=4', 'exp_k_8': 'k=8', 'exp_k_16': 'k=16', 'exp_k_24': 'k=24'}
title = r"Efficiency trade-off: neighborhood size $k$"
output = "figures/generated_figures/pareto_k.pdf"

plot_pareto_efficiency(df, k_runs, k_labels, title, output)