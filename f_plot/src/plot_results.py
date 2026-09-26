import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def summarize(one, paths, validation):
    scores = one.groupby("config")[["nll", "crps", "mae_log", "cover90", "tail_brier",
                                     "gate_variance_share"]].mean()
    scores = scores.join(paths.groupby("config")[["energy", "variogram"]].mean())
    scores = scores.join(validation.groupby("config").val_nll.agg(["mean", "std"]).rename(
        columns={"mean": "val_nll", "std": "val_sd"}))
    return scores.sort_values("nll")

def figures(one, scores, output):
    sns.set_theme(style="whitegrid", context="paper")
    palette = sns.color_palette("colorblind")
    chosen = scores.head(5).index.tolist()
    anchors = ["D"*n for n in (1, 2, 3) if "D"*n in scores.index]
    chosen = list(dict.fromkeys(chosen+anchors))
    yearly = one[one.config.isin(chosen)].groupby(["config", "year"], as_index=False).nll.mean()
    figure, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(yearly, x="year", y="nll", hue="config", style="config", markers=True,
                 dashes=True, palette=palette[:len(chosen)], ax=ax)
    ax.set(xlabel="Target year", ylabel="Predictive NLL", title="Temporal holdout")
    figure.tight_layout()
    figure.savefig(output/"temporal_nll.png", dpi=180)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.barplot(x=scores.index, y=scores.nll, hue=scores.index, palette="colorblind",
                dodge=False, legend=False, ax=axes[0])
    sns.barplot(x=scores.index, y=scores.energy, hue=scores.index, palette="colorblind",
                dodge=False, legend=False, ax=axes[1])
    axes[0].set(ylabel="NLL", title="One-year predictive score")
    axes[1].set(ylabel="Energy score", title="Three-year joint path score")
    for ax in axes: ax.tick_params(axis="x", labelrotation=75)
    figure.tight_layout()
    figure.savefig(output/"score_comparison.png", dpi=180)
    plt.close(figure)

def report(cfg, scores, contrasts, urn, output):
    lines = ["# Stochastic-memory LSTM PDP report", "",
             "All architectures share conditional gate means and a Gaussian readout.",
             "F samples the forget gate, I the input gate, A the input/forget/output gates.",
             f"The seed is {cfg['seed']} throughout; replicates advance its random stream.", "",
             "## Cedar Creek E001 locked temporal holdout", "", "```", scores.to_string(), "```", "",
             "## Paired contrasts to deterministic stacks", "", "```",
             contrasts[["contender", "baseline", "metric", "delta", "low95", "high95", "plots"]].to_string(index=False),
             "```", "", "## Pólya urn trajectory-disjoint holdout", "", "```",
             urn.groupby("config")[["test_nll", "path_energy", "path_variogram"]].agg(["mean", "std"]).to_string(),
             "```", "", "Negative deltas favor a challenger. Repeated annual windows from one plot are dependent;",
             "paired intervals resample whole plots. The E001 series is short for deep models,",
             "and a change in burn regime occurred in 2005. These finite observations cannot prove",
             "non-ergodicity. The Pólya urn supplies a known path-dependent counterpoint, not",
             "independent ecological replication. Check gate_precision.csv for variance collapse,",
             "order_sensitivity.csv for shuffled-history effects, and per-year test rows before",
             "claiming an advantage. The Gaussian biomass head and unequal precision-parameter",
             "counts need sensitivity analyses before generalization."]
    (output/"report.md").write_text("\n".join(lines)+"\n")
