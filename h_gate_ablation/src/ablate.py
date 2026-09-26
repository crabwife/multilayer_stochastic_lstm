"""Paired inference intervention on saved stochastic-gate checkpoints."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import evaluate_one
from processes import generate
from paired import METRICS, bootstrap_contrasts, paired_rows


def run(cfg, checkpoint_output, output):
    checkpoint_output, output = Path(checkpoint_output), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    setting = cfg["nonergodic"]
    data = generate(setting, cfg["seed"])
    missing = [str(checkpoint_output / "checkpoints" / process /
                   f"{config}_rep{rep}.pt")
               for process in setting["processes"]
               for config in setting["configs"]
               for rep in range(setting["repeats"])
               if not (checkpoint_output / "checkpoints" / process /
                       f"{config}_rep{rep}.pt").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} saved benchmark checkpoint(s), first: {missing[0]}. "
            "Run make g_nonergodic first.")
    (output / "manifest.json").write_text(json.dumps({
        "seed": cfg["seed"], "nonergodic_settings": setting,
        "intervention": "Beta gate draws on versus conditional gate means, same fitted weights",
        "emissions": "paired independent Bernoulli uniform stream within each checkpoint and batch",
        "contrast": "on minus off; bootstrap trajectories, training replicates held fixed",
        "checkpoints": str(checkpoint_output)
    }, indent=2))
    all_pairs = []
    for process, split in data.items():
        for config in setting["configs"]:
            for rep in range(setting["repeats"]):
                # This stream is independent of the Beta gate stream and
                # identical for the two evaluations of this checkpoint.
                emission_seed = cfg["seed"] + 100_003 * rep + 503
                on, _ = evaluate_one(cfg, process, split, config, rep,
                                     checkpoint_output, sample_gates=True,
                                     emission_seed=emission_seed)
                off, _ = evaluate_one(cfg, process, split, config, rep,
                                      checkpoint_output, sample_gates=False,
                                      emission_seed=emission_seed)
                pair = paired_rows(on, off)
                if set(config) == {"D"}:
                    # Detect an accidental emission-stream or state mismatch.
                    for metric in METRICS:
                        if not np.allclose(pair[f"delta_{metric}"], 0, atol=1e-12):
                            raise AssertionError(f"Deterministic control changed: {config} {metric}")
                all_pairs.append(pair)
                pd.concat(all_pairs, ignore_index=True).to_csv(
                    output / "trajectory_scores.csv", index=False)
    scores = pd.concat(all_pairs, ignore_index=True)
    contrast = bootstrap_contrasts(scores, setting["bootstrap"], cfg["seed"])
    contrast.to_csv(output / "paired_contrasts.csv", index=False)
    summary = scores.groupby(["process", "config"], as_index=False).agg(
        crps_on=("crps_on", "mean"), crps_off=("crps_off", "mean"),
        delta_crps=("delta_crps", "mean"),
        one_nll_on=("one_nll_on", "mean"), one_nll_off=("one_nll_off", "mean"),
        variance_on=("frequency_variance_on", "mean"),
        variance_off=("frequency_variance_off", "mean"),
        delta_variance=("delta_frequency_variance", "mean"),
        oracle_crps=("oracle_crps_on", "mean"),
        oracle_variance=("oracle_variance_on", "mean"),
        coverage_on=("cover90_on", "mean"), coverage_off=("cover90_off", "mean"))
    summary.to_csv(output / "summary.csv", index=False)
    return summary
