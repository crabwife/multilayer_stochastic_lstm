"""Trajectory-disjoint binary processes and known predictive reference laws."""

import numpy as np
from scipy.stats import betabinom, binom

PROCESSES = ("polya", "iid", "sticky")


def generate(settings, seed):
    """Independent trajectories, split before any training windows are made."""
    context, horizon = settings["context"], settings["horizon"]
    if context < 2 or horizon < 1:
        raise ValueError("Require context >= 2 and horizon >= 1")
    length = max(context + horizon, 3 * context + 1)
    n = sum(settings[f"{s}_trajectories"] for s in ("train", "val", "test"))
    if min(settings[f"{s}_trajectories"] for s in ("train", "val", "test")) < 1:
        raise ValueError("Each split needs an independent trajectory")
    stay = float(settings["sticky_stay"])
    if not 0.5 < stay < 1:
        raise ValueError("Sticky control requires 0.5 < stay < 1")
    result = {}
    for index, process in enumerate(settings["processes"]):
        if process not in PROCESSES:
            raise ValueError(f"Unknown process {process}")
        rng = np.random.default_rng(seed + index * 100_003)
        x = np.empty((n, length), dtype=np.float32)
        if process == "polya":
            # Equivalent in joint law to p ~ Beta(1,1), then iid Bernoulli(p).
            red = np.ones(n)
            total = np.full(n, 2.)
            for t in range(length):
                x[:, t] = rng.random(n) < red / total
                red += x[:, t]
                total += 1
        elif process == "iid":
            x[:] = (rng.random((n, length)) < .5)
        else:
            x[:, 0] = rng.integers(0, 2, n)
            for t in range(1, length):
                flip = rng.random(n) >= stay
                x[:, t] = np.where(flip, 1 - x[:, t-1], x[:, t-1])
        a = settings["train_trajectories"]
        b = a + settings["val_trajectories"]
        result[process] = {"train": x[:a], "val": x[a:b], "test": x[b:]}
    return result


def one_step_windows(trajectories, context):
    """Three nonoverlapping contexts per trajectory; split was already fixed."""
    starts = (0, context, 2 * context)
    if trajectories.shape[1] <= starts[-1] + context:
        raise ValueError("Trajectories are too short for training contexts")
    histories = np.concatenate([trajectories[:, s:s+context] for s in starts])
    targets = np.concatenate([trajectories[:, s+context] for s in starts])
    return histories[:, :, None].copy(), targets.copy()


def sticky_count_pmf(horizon, last, stay):
    """Exact future count distribution for a stationary two-state Markov chain."""
    dp = np.zeros((2, horizon + 1))
    dp[int(last), 0] = 1.
    for step in range(horizon):
        nxt = np.zeros_like(dp)
        for state in (0, 1):
            for state_next in (0, 1):
                prob = stay if state == state_next else 1 - stay
                nxt[state_next, state_next:state_next + step + 1] += prob * dp[state, :step + 1]
        dp = nxt
    return dp.sum(axis=0)


def oracle_pmf(process, history, horizon, stay):
    """Count of ones in the next block, conditional only on observed history."""
    k = np.arange(horizon + 1)
    if process == "polya":
        red = 1 + int(np.sum(history))
        blue = 1 + len(history) - int(np.sum(history))
        return betabinom.pmf(k, horizon, red, blue)
    if process == "iid":
        return binom.pmf(k, horizon, .5)
    if process == "sticky":
        return sticky_count_pmf(horizon, int(history[-1]), stay)
    raise ValueError(process)


def discrete_crps(pmf, observed_count):
    """Exact CRPS on the block-frequency scale (units 1 / horizon)."""
    h = len(pmf) - 1
    cdf = np.cumsum(pmf)[:-1]
    return float(np.sum((cdf - (np.arange(h) >= observed_count)) ** 2) / h)


def ensemble_crps(counts, observed_count, horizon):
    pmf = np.bincount(counts.astype(int), minlength=horizon + 1) / len(counts)
    return discrete_crps(pmf, observed_count)


def interval_coverage(pmf, observed_count):
    cdf = np.cumsum(pmf)
    lo = int(np.searchsorted(cdf, .05))
    hi = int(np.searchsorted(cdf, .95))
    return int(lo <= observed_count <= hi)
