"""
profiler.py
Custom implementation.

Author: Alexandre Devaux Rivière
Project: NPM3D
Date: 20/03/2026
"""

import time
import torch
import csv
import os
from collections import defaultdict


# operation names
OPS = [
    "QKV encoding",
    "KNN query",
    "Relative positional encoding",
    "Relation & weight encoding",
    "Value aggregation",
    "FPS & downsample",
]

# encoder stage names in forward-pass order
STAGES = ["enc1", "enc2", "enc3", "enc4", "enc5"]


class Profiler:
    """
    Latency profiler for PointTransformer.

    Design choices
    --------------
    - Per-stage bucketing: times are stored as times[stage][op], so enc1's
      KNN query is separated from enc5's KNN query in the treemap.
    - Warm-up guard: the first `warmup_batches` batches are skipped
      so that CUDA JIT / kernel-cache effects don't inflate the numbers.
    - Thread-safe accumulation: a threading.Lock guards the dict so that
      DataParallel replica threads don't race on writes.
    - cuda.synchronize() called before both start() and end() to
      ensure the GPU has actually finished work before the wall-clock is read.
    - enabled flag: set to True only during validation (inference), so
      training overhead is zero.
    """

    def __init__(self, warmup_batches: int = 2):
        self.warmup_batches = warmup_batches
        self.enabled = False
        self._batch_count = 0
        self._in_warmup = True
        self.reset()

        import threading
        self._lock = threading.Lock()

    def reset(self):
        self.times = {
            stage: {op: 0.0 for op in OPS} for stage in STAGES
        }
        self._batch_count = 0
        self._in_warmup = True

    def on_batch_start(self):
        """
        Call once at the beginning of each validation batch.
        Handles the warm-up window internally.
        """
        if not self.enabled:
            return
        self._batch_count += 1
        self._in_warmup = self._batch_count <= self.warmup_batches

    def start(self) -> float:
        """
        Record the start of a timed region.
        Returns None if profiling disabled / still in warm-up.
        """
        if not self.enabled or self._in_warmup:
            return None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    def end(self, start_time: float, stage: str, op: str):
        """
        Record the end of a timed region and accumulate the delta (ms).

        Parameters
        ----------
        start_time : value returned by start(); if None, this is a no-op.
        stage : one of STAGES
        op : one of OPS
        """
        if not self.enabled or start_time is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start_time) * 1_000

        with self._lock:
            self.times[stage][op] += elapsed_ms

    def log_summary(self, logger=None):
        """Pretty print a per-stage, per-op breakdown to the logger """
        lines = ["[Profiler] Latency breakdown (ms, cumulative over eval set):"]
        for stage in STAGES:
            stage_total = sum(self.times[stage].values())
            lines.append(f"  {stage}  (total {stage_total:.1f} ms)")
            for op in OPS:
                v = self.times[stage][op]
                if v > 0:
                    lines.append(f"      {op:<35s} {v:>8.2f} ms")
        msg = "\n".join(lines)
        if logger is not None:
            logger.info(msg)
        else:
            print(msg)

    def save_csv(self, path: str, epoch: int):
        """
        Append one row per epoch to a CSV file.

        Columns: epoch, <stage>/<op> for every (stage, op) pair.
        """
        fieldnames = ["epoch"] + [
            f"{stage}/{op}" for stage in STAGES for op in OPS
        ]
        file_exists = os.path.isfile(path)
        row = {"epoch": epoch}
        for stage in STAGES:
            for op in OPS:
                row[f"{stage}/{op}"] = round(self.times[stage][op], 4)

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


# Module-level singleton
# -> imported by pointtransformer_cls.py and train.py
latency_profiler = Profiler(warmup_batches=2)
