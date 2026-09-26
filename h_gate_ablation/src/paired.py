"""Trajectory-level contrasts for the two inference policies."""

import numpy as np
import pandas as pd

METRICS = ("crps", "one_nll", "frequency_variance", "cover90", "predicted_frequency")
KEYS = ("process", "config", "replicate", "trajectory")


def paired_rows(on, off):
    """Join the same held-out trajectory and fitted checkpoint under both policies."""
    left, right = pd.DataFrame(on), pd.DataFrame(off)
    if left.duplicated(list(KEYS)).any() or right.duplicated(list(KEYS)).any():
        raise ValueError("Duplicate trajectory/configuration/replicate scores")
    pair = left.merge(right, on=list(KEYS), suffixes=("_on", "_off"),
                      validate="one_to_one")
    if len(pair) != len(left) or len(pair) != len(right):
        raise ValueError("A paired trajectory is missing")
    for shared in ("observed_frequency", "oracle_frequency", "oracle_crps",
                   "oracle_variance", "oracle_one_nll"):
        if not np.allclose(pair[f"{shared}_on"], pair[f"{shared}_off"], atol=1e-12):
            raise ValueError(f"Pairing changed the observed or oracle {shared}")
    for metric in METRICS:
        pair[f"delta_{metric}"] = pair[f"{metric}_on"] - pair[f"{metric}_off"]
    return pair


def bootstrap_contrasts(paired, nboot, seed):
    rng = np.random.default_rng(seed + 1013)
    rows = []
    for (process, config), group in paired.groupby(["process", "config"]):
        for metric in METRICS:
            column = f"delta_{metric}"
            by_trajectory = group.groupby("trajectory")[column].mean().to_numpy()
            draws = rng.choice(by_trajectory,
                               size=(nboot, len(by_trajectory)), replace=True).mean(axis=1)
            rows.append({
                "process": process, "config": config, "metric": metric,
                "delta_on_minus_off": float(by_trajectory.mean()),
                "low95": float(np.quantile(draws, .025)),
                "high95": float(np.quantile(draws, .975)),
                "independent_test_trajectories": len(by_trajectory),
                "training_replicates": int(group.replicate.nunique())
            })
    return pd.DataFrame(rows)
