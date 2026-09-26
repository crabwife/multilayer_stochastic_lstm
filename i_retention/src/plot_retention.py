"""Scientific diagnostic of survival and time-average dispersion."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_retention(curves, averages, destination):
    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = dict(zip(sorted(curves.beta.unique()), sns.color_palette("colorblind")))
    for (b, process), sub in curves.groupby(["beta", "process"]):
        sub = sub[sub.steps > 0].sort_values("steps")
        label = f"{process}, b={b:g}"
        sns.lineplot(data=sub, x="steps", y="exact_survival", ax=axes[0],
                     color=colors[b], linestyle="-" if process == "episodic" else "--",
                     label=label)
        observed = sub[sub.empirical_survival > 0]
        sns.scatterplot(data=observed, x="steps", y="empirical_survival", ax=axes[0],
                        color=colors[b], marker="o" if process == "episodic" else "X",
                        s=14, alpha=.45, legend=False)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Steps since episode start",
                ylabel="P(episode survives)", title="Exact laws and simulations")
    axes[0].set_ylim(1e-6, 1.2)
    axes[0].legend(frameon=False, fontsize=8)
    for (b, process), sub in averages.groupby(["beta", "process"]):
        sub = sub.sort_values("length")
        sns.lineplot(data=sub, x="length", y="trajectory_mean_variance", ax=axes[1],
                     marker="o" if process == "episodic" else "X", color=colors[b],
                     linestyle="-" if process == "episodic" else "--",
                     label=f"{process}, b={b:g}")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Trajectory length", ylabel="Variance of time averages",
                title="Finite-horizon dispersion")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=.15)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)
