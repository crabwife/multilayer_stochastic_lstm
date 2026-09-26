"""Matched Beta retention schedules and their exact survival laws."""

import numpy as np
from scipy.special import betaln, digamma, gammaln


def beta_shapes(mean_stay, second_shape):
    if not (0 < mean_stay < 1 and second_shape > 0):
        raise ValueError("Require 0 < mean_stay < 1 and second_shape > 0")
    return second_shape * mean_stay / (1 - mean_stay), second_shape


def survival(steps, mean_stay, second_shape, process):
    """P(initial episode survives more than steps further transitions)."""
    t = np.asarray(steps, dtype=np.float64)
    a, b = beta_shapes(mean_stay, second_shape)
    if process == "annealed":
        return np.power(mean_stay, t)
    if process == "episodic":
        return np.exp(betaln(a + t, b) - betaln(a, b))
    raise ValueError(process)


def retention_diagnostics(mean_stay, second_shape):
    a, b = beta_shapes(mean_stay, second_shape)
    return {
        "alpha": a, "beta": b, "one_step_mean": mean_stay,
        "step_typical_log_retention": float(digamma(a) - digamma(a + b)),
        "deterministic_log_retention": float(np.log(mean_stay)),
        "episodic_tail_prefactor": float(np.exp(gammaln(a + b) - gammaln(a))),
        "episodic_mean_duration": float((a + b - 1) / (b - 1)) if b > 1 else float("inf"),
        "annealed_mean_duration": float(1 / (1 - mean_stay))
    }


def simulate(n, length, mean_stay, second_shape, process, seed):
    """Alternating binary renewal with a Beta stay propensity.

    Annealed: redraw q every step; episodic: draw q at each change of state.
    Both draw q from the same Beta(a,b) at the start of an episode.
    """
    if n < 1 or length < 2:
        raise ValueError("Require n >= 1 and length >= 2")
    a, b = beta_shapes(mean_stay, second_shape)
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 2, n, dtype=np.int8)
    values = np.empty((n, length), dtype=np.float32)
    values[:, 0] = initial
    if process == "annealed":
        q = rng.beta(a, b, size=(n, length - 1))
        flip = rng.random(q.shape) >= q
        values[:, 1:] = (initial[:, None] + np.cumsum(flip, axis=1)) % 2
    elif process == "episodic":
        for j, state in enumerate(initial):
            start = 0
            while start < length:
                q = min(float(rng.beta(a, b)), np.nextafter(1., 0.))
                duration = int(rng.geometric(1 - q))
                stop = min(length, start + duration)
                values[j, start:stop] = state
                start, state = stop, 1 - state
    else:
        raise ValueError(process)
    return values


def generate(settings, seed):
    n = sum(settings[f"{s}_trajectories"] for s in ("train", "val", "test"))
    if min(settings[f"{s}_trajectories"] for s in ("train", "val", "test")) < 1:
        raise ValueError("Each split needs independent trajectories")
    context = settings["context"]
    length = max(context + settings["test_horizon"],
                 3 * context + settings["train_horizon"])
    data = {}
    for index, (b, process) in enumerate(
        (b, process) for b in settings["beta_second_shape"]
        for process in settings["processes"]
    ):
        key = f"{process}_b{b:g}"
        trajectories = simulate(n, length, settings["mean_stay"], b, process,
                                seed + 100_003 * index)
        a = settings["train_trajectories"]
        z = a + settings["val_trajectories"]
        data[key] = {"train": trajectories[:a], "val": trajectories[a:z],
                     "test": trajectories[z:], "true_process": process, "true_beta": b}
    return data


def training_windows(trajectories, context, horizon):
    starts = (0, context, 2 * context)
    if trajectories.shape[1] < starts[-1] + context + horizon:
        raise ValueError("Insufficient trajectory length")
    x = np.concatenate([trajectories[:, s:s + context] for s in starts])
    y = np.concatenate([trajectories[:, s + context:s + context + horizon]
                        for s in starts])
    return x[:, :, None].copy(), y.copy()


def theory_tables(settings, seed):
    import pandas as pd

    times = np.unique(np.r_[0, np.geomspace(1, settings["theory_horizon"] - 1,
                                            num=35).astype(int)])
    curves, averages = [], []
    for index, b in enumerate(settings["beta_second_shape"]):
        diag = retention_diagnostics(settings["mean_stay"], b)
        for j, process in enumerate(settings["processes"]):
            values = simulate(settings["theory_trajectories"],
                              settings["theory_horizon"], settings["mean_stay"],
                              b, process, seed + 900_001 + index * 101 + j)
            first_flip = np.where((values[:, 1:] != values[:, [0]]).any(axis=1),
                                  (values[:, 1:] != values[:, [0]]).argmax(axis=1) + 1,
                                  values.shape[1])
            for t in times:
                curves.append({
                    "process": process, "beta": b, "steps": int(t),
                    "exact_survival": float(survival(t, settings["mean_stay"], b, process)),
                    "empirical_survival": float(np.mean(first_flip > t)),
                    **diag
                })
            for length in (128, 512, settings["theory_horizon"]):
                if length > values.shape[1]:
                    continue
                means = values[:, :length].mean(axis=1)
                averages.append({
                    "process": process, "beta": b, "length": length,
                    "trajectory_mean_variance": float(means.var()),
                    "trajectory_mean_q05": float(np.quantile(means, .05)),
                    "trajectory_mean_q50": float(np.quantile(means, .5)),
                    "trajectory_mean_q95": float(np.quantile(means, .95)),
                    "trajectories": len(means)
                })
    return pd.DataFrame(curves), pd.DataFrame(averages)
