"""Joint-path fitting and longer-horizon evaluation of Beta forget schedules."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from processes import ensemble_crps, interval_coverage
from retention_data import generate, theory_tables, training_windows
from retention_model import RetentionLSTM
from plot_retention import plot_retention


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _signature(cfg, key, split):
    prior_cfg = {name: value for name, value in cfg.items()
                 if name != "layered_retention"}
    h = hashlib.sha256(json.dumps(prior_cfg, sort_keys=True).encode() + key.encode())
    for name in ("train", "val"):
        h.update(split[name].tobytes())
    return h.hexdigest()


def _loaders(split, setting, seed):
    tensors = []
    for name in ("train", "val"):
        x, y = training_windows(split[name], setting["context"],
                                setting["train_horizon"])
        tensors.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)))
    train = DataLoader(tensors[0], batch_size=setting["batch"], shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(tensors[1], batch_size=setting["batch"], shuffle=False)
    return train, val


def fit_one(cfg, key, split, policy, replicate, output):
    setting = cfg["retention"]
    file = output / "checkpoints" / key / f"{policy}_rep{replicate}.pt"
    file.parent.mkdir(parents=True, exist_ok=True)
    signature = _signature(cfg, key, split)
    if file.exists():
        saved = torch.load(file, map_location="cpu", weights_only=True)
        if saved["signature"] != signature:
            raise ValueError(f"Stale checkpoint {file}; clear i_retention/output/checkpoints")
        return saved["summary"]
    seed = cfg["seed"] + replicate * 10_007
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    model = RetentionLSTM(setting["width"], policy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting["lr"],
                                  weight_decay=setting["weight_decay"])
    train, val = _loaders(split, setting, seed)
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.time()
    for epoch in range(1, setting["epochs"] + 1):
        model.train()
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            loss = model.path_nll(x.to(device), y.to(device),
                                  setting["particles_train"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        score, n = 0., 0
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 301)
            for x, y in val:
                loss = model.path_nll(x.to(device), y.to(device),
                                      setting["particles_val"])
                score += loss.item() * len(y)
                n += len(y)
        score /= n
        if score < best - 1e-4:
            best, best_epoch = score, epoch
            best_state = {name: p.detach().cpu().clone()
                          for name, p in model.state_dict().items()}
        if epoch - best_epoch >= setting["patience"]:
            break
    summary = {"process": key, "policy": policy, "replicate": replicate,
               "val_path_nll": best, "epoch": best_epoch,
               "params": sum(p.numel() for p in model.parameters()),
               "seconds": time.time() - started}
    torch.save({"signature": signature, "state": best_state, "summary": summary}, file)
    return summary


def evaluate_one(cfg, key, split, trained_policy, inference_policy, replicate, output):
    setting = cfg["retention"]
    file = output / "checkpoints" / key / f"{trained_policy}_rep{replicate}.pt"
    saved = torch.load(file, map_location="cpu", weights_only=True)
    if saved["signature"] != _signature(cfg, key, split):
        raise ValueError(f"Stale checkpoint {file}")
    device = _device()
    model = RetentionLSTM(setting["width"], trained_policy).to(device)
    model.load_state_dict(saved["state"])
    model.policy = inference_policy  # Same fitted weights, different refresh schedule.
    model.eval()
    torch.manual_seed(cfg["seed"] + replicate * 10_007 + 501)
    history = split["test"][:, :setting["context"], None]
    targets = split["test"][:, setting["context"]:
                            setting["context"] + setting["test_horizon"]]
    rows = []
    with torch.no_grad():
        for start in range(0, len(history), setting["batch"]):
            x = torch.from_numpy(history[start:start + setting["batch"]]).to(device)
            sampled = model.sample_paths(
                x, setting["particles_test"], setting["test_horizon"],
                emission_seed=cfg["seed"] + replicate * 10_007 + 13_009 + start
            ).cpu().numpy()
            for j, paths in enumerate(sampled):
                index = start + j
                for h in setting["evaluation_horizons"]:
                    counts = paths[:, :h].sum(axis=-1)
                    observed_count = int(targets[index, :h].sum())
                    pmf = np.bincount(counts.astype(int), minlength=h + 1) / len(counts)
                    rows.append({
                        "process": key, "true_schedule": split["true_process"],
                        "true_beta": split["true_beta"], "policy": inference_policy,
                        "trained_policy": trained_policy,
                        "replicate": replicate, "trajectory": index, "horizon": h,
                        "observed_frequency": observed_count / h,
                        "predicted_frequency": float(counts.mean() / h),
                        "frequency_variance": float(counts.var() / h**2),
                        "crps": ensemble_crps(counts, observed_count, h),
                        "cover90": interval_coverage(pmf, observed_count)
                    })
    diagnostics = {"process": key, "policy": trained_policy, "replicate": replicate,
                   **model.retention_summary(torch.from_numpy(history).to(device))}
    return rows, diagnostics


def paired_contrasts(scores, nboot, seed):
    rng = np.random.default_rng(seed + 901)
    rows = []
    for (process, horizon), subset in scores.groupby(["process", "horizon"]):
        for contender, baseline in (("step", "mean"), ("episode", "mean"),
                                    ("episode", "step")):
            working = subset[subset.policy.isin((contender, baseline))]
            for metric in ("crps", "frequency_variance", "cover90"):
                wide = working.pivot(index=["trajectory", "replicate"],
                                     columns="policy", values=metric)
                if wide.isna().any().any():
                    raise ValueError(f"Missing paired {process} {metric}")
                differences = (wide[contender] - wide[baseline]).groupby(
                    "trajectory").mean().to_numpy()
                draws = rng.choice(differences,
                                   size=(nboot, len(differences)),
                                   replace=True).mean(axis=1)
                rows.append({
                    "process": process, "horizon": horizon,
                    "contender": contender, "baseline": baseline,
                    "metric": metric, "delta": float(differences.mean()),
                    "low95": float(np.quantile(draws, .025)),
                    "high95": float(np.quantile(draws, .975)),
                    "independent_trajectories": len(differences),
                    "training_replicates": int(subset.replicate.nunique())
                })
    return pd.DataFrame(rows)


def paired_interventions(scores, nboot, seed):
    """Within-checkpoint refresh-policy contrasts, with paired emissions."""
    rng = np.random.default_rng(seed + 1109)
    rows = []
    for (process, horizon, trained), subset in scores.groupby(
        ["process", "horizon", "trained_policy"]
    ):
        for contender, baseline in (("step", "mean"), ("episode", "mean"),
                                    ("episode", "step")):
            working = subset[subset.policy.isin((contender, baseline))]
            for metric in ("crps", "frequency_variance", "cover90"):
                wide = working.pivot(index=["trajectory", "replicate"],
                                     columns="policy", values=metric)
                if wide.isna().any().any():
                    raise ValueError(f"Missing crossover for {process} {trained}")
                differences = (wide[contender] - wide[baseline]).groupby(
                    "trajectory").mean().to_numpy()
                draws = rng.choice(differences, size=(nboot, len(differences)),
                                   replace=True).mean(axis=1)
                rows.append({
                    "process": process, "horizon": horizon, "trained_policy": trained,
                    "contender": contender, "baseline": baseline, "metric": metric,
                    "delta": float(differences.mean()),
                    "low95": float(np.quantile(draws, .025)),
                    "high95": float(np.quantile(draws, .975)),
                    "independent_trajectories": len(differences),
                    "training_replicates": int(subset.replicate.nunique())
                })
    return pd.DataFrame(rows)


def run(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    setting = cfg["retention"]
    data = generate(setting, cfg["seed"])
    theory, averages = theory_tables(setting, cfg["seed"])
    theory.to_csv(output / "retention_theory.csv", index=False)
    averages.to_csv(output / "time_average.csv", index=False)
    plot_retention(theory, averages, output / "retention_laws.png")
    (output / "manifest.json").write_text(json.dumps({
        "seed": cfg["seed"], "settings": setting,
        "split": "independent trajectories per process; 3 fixed training contexts",
        "selection": "best 16-step validation path likelihood per model; test never selects",
        "parameter_matching": "all policies use identical layers and parameter count",
        "identification": "analytic survival is for the controlled process, not a theorem about the trained model"
    }, indent=2))
    validation, rows, cross_rows, diagnostics = [], [], [], []
    for key, split in data.items():
        for trained_policy in setting["policies"]:
            for rep in range(setting["repeats"]):
                validation.append(fit_one(cfg, key, split, trained_policy, rep, output))
                for inference_policy in setting["policies"]:
                    result, detail = evaluate_one(
                        cfg, key, split, trained_policy, inference_policy, rep, output)
                    cross_rows.extend(result)
                    if inference_policy == trained_policy:
                        rows.extend(result)
                        diagnostics.append(detail)
                pd.DataFrame(validation).to_csv(output / "validation.csv", index=False)
                pd.DataFrame(rows).to_csv(output / "trajectory_scores.csv", index=False)
                pd.DataFrame(cross_rows).to_csv(
                    output / "schedule_interventions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output / "learned_retention.csv", index=False)
    scores = pd.DataFrame(rows)
    paired_contrasts(scores, setting["bootstrap"], cfg["seed"]).to_csv(
        output / "paired_contrasts.csv", index=False)
    cross = pd.DataFrame(cross_rows)
    paired_interventions(cross, setting["bootstrap"], cfg["seed"]).to_csv(
        output / "paired_interventions.csv", index=False)
    intervention_summary = cross.groupby(
        ["process", "true_schedule", "true_beta", "trained_policy", "policy", "horizon"],
        as_index=False).agg(
        crps=("crps", "mean"), cover90=("cover90", "mean"),
        frequency_variance=("frequency_variance", "mean"))
    intervention_summary.to_csv(output / "schedule_intervention_summary.csv", index=False)
    summary = scores.groupby(["process", "true_schedule", "true_beta", "policy", "horizon"],
                             as_index=False).agg(
        crps=("crps", "mean"), cover90=("cover90", "mean"),
        frequency_variance=("frequency_variance", "mean"),
        predicted_frequency=("predicted_frequency", "mean"),
        observed_frequency=("observed_frequency", "mean"))
    summary.to_csv(output / "summary.csv", index=False)
    return summary
