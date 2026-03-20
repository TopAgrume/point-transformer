"""
extract_tensorboard_data.py
Custom implementation.

Author: Alexandre Devaux Rivière
Project: NPM3D
Date: 20/03/2026
"""

import os
from pathlib import Path
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

root = Path("all_exp/")

all_rows = []

for exp_dir in tqdm(root.glob("exp_*")):
    event_files = list(exp_dir.glob("**/events.out.tfevents.*"))

    for event_file in event_files:
        ea = EventAccumulator(str(event_file))
        ea.Reload()

        run_name = exp_dir.name

        for tag in ea.Tags()["scalars"]:
            if tag in ["acc_train_batch", "grad_norm_before_clip", "loss_train_batch"]:
                continue # skip useless debug metrics

            events = ea.Scalars(tag)

            for e in events:
                all_rows.append({
                    "run": run_name,
                    "tag": tag,
                    "step": e.step,
                    "value": e.value,
                    "wall_time": e.wall_time
                })

df = pd.DataFrame(all_rows)

df.to_csv("figures/csv/tensorboard_all_runs.csv", index=False)