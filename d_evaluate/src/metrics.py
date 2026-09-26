"""Proper and descriptive predictive scores; no training dependencies."""

import math

import numpy as np
from scipy.special import logsumexp


def gaussian_mixture_nll(means, scales, target):
    logp = -.5 * (((target[:, None] - means) / scales) ** 2
                  + 2 * np.log(scales) + math.log(2 * math.pi))
    return -(logsumexp(logp, axis=1) - math.log(means.shape[1]))


def sample_crps(samples, target):
    """Empirical CRPS with the exact empirical pairwise term, O(P log P)."""
    samples = np.sort(samples, axis=1)
    p = samples.shape[1]
    rank = np.arange(1, p + 1)
    pair_term = np.sum(samples * (2 * rank - p - 1), axis=1) / (p * p)
    return np.mean(np.abs(samples - target[:, None]), axis=1) - pair_term


def energy_score(paths, observed):
    """Energy score for joint H-step predictive paths; lower is better."""
    first = np.linalg.norm(paths - observed[:, None, :], axis=-1).mean(axis=1)
    pair = np.linalg.norm(paths[:, :, None, :] - paths[:, None, :, :], axis=-1)
    return first - .5 * pair.mean(axis=(1, 2))


def variogram_score(paths, observed, power=.5):
    h = observed.shape[1]
    result = np.zeros(len(observed))
    for j in range(h):
        for k in range(j + 1, h):
            model = np.abs(paths[:, :, j] - paths[:, :, k]) ** power
            truth = np.abs(observed[:, j] - observed[:, k]) ** power
            result += (model.mean(axis=1) - truth) ** 2
    return result


def paired_plot_bootstrap(rows, contender, baseline, metric, nboot=1000, seed=2026):
    """Resample entire plots; keep repeated target years and seeds together."""
    sub = rows[rows.config.isin([contender, baseline])]
    wide = sub.pivot_table(index=["seed", "plot_id", "year"],
                           columns="config", values=metric)
    wide = wide.dropna(subset=[contender, baseline]).reset_index()
    wide["difference"] = wide[contender] - wide[baseline]
    byplot = wide.groupby("plot_id").difference.agg(["mean", "size"])
    ids = np.arange(len(byplot))
    rng = np.random.default_rng(seed)
    vals = byplot["mean"].to_numpy()
    weights = byplot["size"].to_numpy()
    estimates = []
    for _ in range(nboot):
        draw = rng.choice(ids, size=len(ids), replace=True)
        estimates.append(float(np.average(vals[draw], weights=weights[draw])))
    return {"contender": contender, "baseline": baseline, "metric": metric,
            "delta": float(wide.difference.mean()),
            "low95": float(np.quantile(estimates, .025)),
            "high95": float(np.quantile(estimates, .975)),
            "seed_deltas": wide.groupby("seed").difference.mean().round(5).to_dict(),
            "plots": len(byplot)}
