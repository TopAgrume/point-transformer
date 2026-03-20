"""
graphs_generation_for_report.py
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
plt.rcParams.update({
    "figure.autolayout": True,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

def load_and_preprocess_data(csv_filepath):
    df = pd.read_csv(csv_filepath, names=["run", "tag", "step", "value", "wall_time"], header=0)
    return df

def generate_summary_table(df):
    """accuracy Max and total time for each run"""
    summary = []
    runs = df['run'].unique()

    for run in runs:
        run_data = df[df['run'] == run]

        # val metrics extraction
        oa_val_data = run_data[run_data['tag'] == 'allAcc_val']
        macc_val_data = run_data[run_data['tag'] == 'mAcc_val']

        max_oa = oa_val_data['value'].max() if not oa_val_data.empty else np.nan
        max_macc = macc_val_data['value'].max() if not macc_val_data.empty else np.nan

        # total training time (h)
        if not run_data.empty:
            min_time = run_data['wall_time'].min()
            max_time = run_data['wall_time'].max()
            total_time_hours = (max_time - min_time) / 3600
        else:
            total_time_hours = np.nan

        summary.append({
            "Experiment": run,
            "Max_OA": max_oa * 100 if pd.notnull(max_oa) else np.nan,
            "Max_mean_acc": max_macc * 100 if pd.notnull(max_macc) else np.nan,
            "Total_time": total_time_hours
        })

    summary_df = pd.DataFrame(summary).sort_values(by="Max_OA", ascending=False)
    summary_df.to_csv("figures/csv/ablation_summary_table.csv", index=False, float_format="%.2f")
    print(summary_df)
    return summary_df

def plot_ablation_study(df, study_name, runs_to_include, labels_dict, filename):
    """accuracy evolution for ablation sutdy"""
    plt.figure(figsize=(8, 5))
    smoothing=5
    # filter specific run and validation tag
    study_data = df[(df['run'].isin(runs_to_include)) & (df['tag'] == 'allAcc_val')].copy()

    study_data['run_label'] = study_data['run'].map(labels_dict)

    study_data = study_data.sort_values('step')
    study_data['value_smooth'] = (
        study_data.groupby('run_label')['value']
        .transform(lambda x: x.rolling(smoothing, min_periods=1, center=True).mean())
    )

    # curve
    ax = sns.lineplot(
        data=study_data,
        x="step",
        y="value",
        hue="run_label",
        linewidth=2,
        alpha=0.8
    )

    plt.title(f"Ablation study: {study_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Overall accuracy (test)")
    plt.ylim(0.75, 0.95)
    plt.legend(title="Configuration:", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(filename, format='pdf', dpi=300)
    plt.close()
    print(f"Graphique généré : {filename}")

def main():
    csv_path = "figures/csv/tensorboard_all_runs.csv"

    try:
        df = load_and_preprocess_data(csv_path)
    except FileNotFoundError:
        print(f"{csv_path} not found.")
        return

    # Tables
    generate_summary_table(df)

    # ablation studies graphs:
    # Neighborhood size
    k_runs = ['exp_k_4', 'exp_k_8', 'exp_k_16', 'exp_k_24']
    k_labels = {
        'exp_k_4': r'$k = 4$',
        'exp_k_8': r'$k = 8$ (best)',
        'exp_k_16': r'$k = 16$ (baseline)',
        'exp_k_24': r'$k = 24$'
    }
    plot_ablation_study(df, r"Neighborhood size $k$", k_runs, k_labels, "figures/generated_figures/ablation_k.pdf")

    # Attention strat
    att_runs = ['exp_k_8', 'exp_dot_product', 'exp_mlp', 'exp_mlp_pooling']
    att_labels = {
        'exp_k_8': r'Vector attention ($k=8$)',
        'exp_dot_product': r'Scalar attention ($k=8$)',
        'exp_mlp_pooling': 'MLP + pooling',
        'exp_mlp': 'MLP',
    }
    plot_ablation_study(df, "Attention mechanism", att_runs, att_labels, "figures/generated_figures/ablation_attention.pdf")

    # Positional encoding
    pos_runs = ['exp_k_8', 'exp_absolute', 'exp_magnitude', 'exp_none']
    pos_labels = {
        'exp_k_8': 'Relative (best)',
        'exp_absolute': 'Absolute',
        'exp_magnitude': 'Magnitude',
        'exp_none': 'None'
    }
    plot_ablation_study(df, "Positional encoding", pos_runs, pos_labels, "figures/generated_figures/ablation_pos_enc.pdf")

    # Optimizers & schedulers
    opt_runs = ['exp_k_8', 'exp_adamw_oncycle_clip', 'exp_adamw_onecycle_no_clip', 'exp_madgrad_cosine', 'exp_sgd_onecycle']
    opt_labels = {
        'exp_k_8': 'SGD + MultiStep (best)',
        'exp_adamw_oncycle_clip': 'AdamW + OneCycle (clip)',
        'exp_adamw_onecycle_no_clip': 'AdamW + OneCycle (no clip)',
        'exp_madgrad_cosine': 'MADGRAD + Cosine',
        'exp_sgd_onecycle': 'SGD + OneCycle'
    }
    plot_ablation_study(df, "Optimizers & Schedulers", opt_runs, opt_labels, "figures/generated_figures/ablation_optimizers.pdf")

if __name__ == "__main__":
    main()