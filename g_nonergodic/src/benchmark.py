"""Fitted gate models, locked long-horizon evaluation, and paired comparisons."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from binary_model import BinaryGateLSTM
from processes import (discrete_crps, ensemble_crps, generate,
                       interval_coverage, one_step_windows, oracle_pmf)


def _signature(cfg, process, split):
    # A new independent retention task must not invalidate existing synthetic fits.
    benchmark_cfg = {key: value for key, value in cfg.items()
                     if key not in ("retention", "layered_retention")}
    digest = hashlib.sha256(json.dumps(benchmark_cfg, sort_keys=True).encode() + process.encode())
    for key in ("train", "val"):
        digest.update(split[key].tobytes())
    return digest.hexdigest()


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _loss(probability, target):
    return torch.nn.functional.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6), target)


def fit_one(cfg, process, split, config, replicate, output):
    setting = cfg["nonergodic"]
    seed = cfg["seed"] + 10_007 * replicate
    file = output / "checkpoints" / process / f"{config}_rep{replicate}.pt"
    file.parent.mkdir(parents=True, exist_ok=True)
    signature = _signature(cfg, process, split)
    if file.exists():
        state = torch.load(file, map_location="cpu", weights_only=True)
        if state["signature"] != signature:
            raise ValueError(f"Stale checkpoint {file}; clear g_nonergodic/output/checkpoints")
        return state["summary"]

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device()
    model = BinaryGateLSTM(config, cfg["model"]["width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["fit"]["lr"],
                                  weight_decay=cfg["fit"]["weight_decay"])
    context = setting["context"]
    tx, ty = one_step_windows(split["train"], context)
    vx, vy = one_step_windows(split["val"], context)
    train = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
                       batch_size=setting["batch"], shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    val = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
                     batch_size=setting["batch"], shuffle=False)
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.time()
    for epoch in range(1, setting["epochs"] + 1):
        model.train()
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            p = model(x.to(device), setting["particles_train"])
            loss = _loss(p, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        score, n = 0., 0
        devices = list(range(torch.cuda.device_count())) if device.type == "cuda" else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 301)
            for x, y in val:
                loss = _loss(model(x.to(device), setting["particles_val"]), y.to(device))
                score += loss.item() * len(y)
                n += len(y)
        score /= n
        if score < best - 1e-4:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= setting["patience"]:
            break
    summary = {"process": process, "config": config, "replicate": replicate,
               "val_nll": best, "epoch": best_epoch, "seconds": time.time() - started,
               "params": sum(p.numel() for p in model.parameters())}
    torch.save({"signature": signature, "state": best_state, "summary": summary}, file)
    return summary


def evaluate_one(cfg, process, split, config, replicate, output,
                 sample_gates=True, emission_seed=None):
    setting = cfg["nonergodic"]
    file = output / "checkpoints" / process / f"{config}_rep{replicate}.pt"
    saved = torch.load(file, map_location="cpu", weights_only=True)
    if saved["signature"] != _signature(cfg, process, split):
        raise ValueError(f"Stale checkpoint {file}")
    model = BinaryGateLSTM(config, cfg["model"]["width"]).to(_device())
    model.load_state_dict(saved["state"])
    model.eval()
    torch.manual_seed(cfg["seed"] + 10_007 * replicate + 501)
    context, horizon = setting["context"], setting["horizon"]
    truth = split["test"]
    x = torch.as_tensor(truth[:, :context, None], device=_device())
    rows = []
    for start in range(0, len(x), setting["batch"]):
        xx = x[start:start + setting["batch"]]
        with torch.no_grad():
            p = model(xx, setting["particles_test"], sample_gates=sample_gates).cpu().numpy()
            counts = model.sample_paths(
                xx, setting["particles_test"], horizon, sample_gates=sample_gates,
                emission_seed=None if emission_seed is None else emission_seed + start
            ).sum(-1).cpu().numpy()
        for j, (prob, ensemble) in enumerate(zip(p, counts)):
            ix = start + j
            history = truth[ix, :context]
            target = truth[ix, context:context + horizon]
            observed_count = int(target.sum())
            oracle = oracle_pmf(process, history, horizon, setting["sticky_stay"])
            oracle_next = (1 + history.sum()) / (2 + context) if process == "polya" else (
                .5 if process == "iid" else
                setting["sticky_stay"] if history[-1] else 1 - setting["sticky_stay"])
            predicted = np.bincount(ensemble.astype(int), minlength=horizon + 1) / len(ensemble)
            one_target = float(target[0])
            rows.append({
                "process": process, "config": config, "replicate": replicate,
                "trajectory": ix, "observed_frequency": observed_count / horizon,
                "predicted_frequency": float(ensemble.mean() / horizon),
                "oracle_frequency": float(np.dot(oracle, np.arange(horizon + 1)) / horizon),
                "frequency_variance": float(ensemble.var() / horizon**2),
                "oracle_variance": float(np.dot(oracle, (np.arange(horizon + 1) / horizon)**2)
                                         - (np.dot(oracle, np.arange(horizon + 1)) / horizon)**2),
                "crps": ensemble_crps(ensemble, observed_count, horizon),
                "oracle_crps": discrete_crps(oracle, observed_count),
                "cover90": interval_coverage(predicted, observed_count),
                "oracle_cover90": interval_coverage(oracle, observed_count),
                "one_nll": float(-np.log(prob if one_target else 1 - prob)),
                "oracle_one_nll": float(-np.log(oracle_next if one_target else 1 - oracle_next))
            })
    return rows, [{"process": process, "config": config, "replicate": replicate, **item}
                  for item in model.precision_summary()]


def contrasts(scores, nboot, seed):
    """Resample trajectories, averaging training seeds within each paired unit."""
    rng = np.random.default_rng(seed + 809)
    rows = []
    for process in sorted(scores.process.unique()):
        subset = scores[scores.process == process]
        for contender in sorted(subset.config.unique()):
            baseline = "D" * len(contender)
            if contender == baseline or baseline not in set(subset.config):
                continue
            for metric in ("crps", "one_nll", "frequency_variance"):
                paired = subset[subset.config.isin((contender, baseline))].pivot(
                    index=["trajectory", "replicate"], columns="config", values=metric)
                if paired.isna().any().any():
                    raise ValueError("Missing paired benchmark scores")
                difference = (paired[contender] - paired[baseline]).groupby("trajectory").mean().to_numpy()
                draws = rng.choice(difference, size=(nboot, len(difference)), replace=True).mean(axis=1)
                rows.append({"process": process, "config": contender, "baseline": baseline,
                             "metric": metric, "delta": float(difference.mean()),
                             "low95": float(np.quantile(draws, .025)),
                             "high95": float(np.quantile(draws, .975)),
                             "independent_test_trajectories": len(difference)})
    return pd.DataFrame(rows)


def run(cfg, output):
    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    setting = cfg["nonergodic"]
    if len(set(setting["configs"])) != len(setting["configs"]):
        raise ValueError("Duplicate benchmark configurations")
    data = generate(setting, cfg["seed"])
    manifest = {
        "seed": cfg["seed"], "settings": setting,
        "meaning": {"polya": "reinforcement; conditional Beta-binomial future count",
                    "iid": "ergodic independent Bernoulli(0.5)",
                    "sticky": "ergodic Markov chain with fixed transition probability"},
        "split": "independent trajectories, no windows shared across splits",
        "model_selection": "best one-step validation NLL; test evaluated once per checkpoint"
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    training, scores, precision = [], [], []
    for process, split in data.items():
        for config in setting["configs"]:
            for rep in range(setting["repeats"]):
                training.append(fit_one(cfg, process, split, config, rep, output))
                row, gate = evaluate_one(cfg, process, split, config, rep, output)
                scores.extend(row)
                precision.extend(gate)
                pd.DataFrame(training).to_csv(output / "validation.csv", index=False)
                pd.DataFrame(scores).to_csv(output / "trajectory_scores.csv", index=False)
    pd.DataFrame(precision).to_csv(output / "gate_precision.csv", index=False)
    frame = pd.DataFrame(scores)
    contrasts(frame, setting["bootstrap"], cfg["seed"]).to_csv(
        output / "paired_contrasts.csv", index=False)
    summary = frame.groupby(["process", "config"], as_index=False).agg(
        crps=("crps", "mean"), oracle_crps=("oracle_crps", "mean"),
        one_nll=("one_nll", "mean"), oracle_one_nll=("oracle_one_nll", "mean"),
        cover90=("cover90", "mean"), oracle_cover90=("oracle_cover90", "mean"),
        frequency_variance=("frequency_variance", "mean"),
        oracle_variance=("oracle_variance", "mean"))
    summary.to_csv(output / "summary.csv", index=False)
    return summary
