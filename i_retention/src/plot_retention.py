"""Scientific diagnostic of survival and time-average dispersion."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_retention(curves, averages, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {b: color for b, color in zip(sorted(curves.beta.unique()),
                                          ("#1877a5", "#b04a32", "#568849"))}
    for (b, process), sub in curves.groupby(["beta", "process"]):
        sub = sub[sub.steps > 0].sort_values("steps")
        label = f"{process}, b={b:g}"
        axes[0].loglog(sub.steps, sub.exact_survival, color=colors[b],
                       linestyle="-" if process == "episodic" else "--",
                       label=label)
        observed = sub[sub.empirical_survival > 0]
        axes[0].scatter(observed.steps, observed.empirical_survival,
                        color=colors[b], marker="o" if process == "episodic" else "x",
                        s=10, alpha=.45)
    axes[0].set(xlabel="Steps since episode start",
                ylabel="P(episode survives)", title="Exact laws and simulations")
    axes[0].set_ylim(1e-6, 1.2)
    axes[0].legend(frameon=False, fontsize=8)
    for (b, process), sub in averages.groupby(["beta", "process"]):
        sub = sub.sort_values("length")
        axes[1].loglog(sub.length, sub.trajectory_mean_variance,
                       marker="o", color=colors[b],
                       linestyle="-" if process == "episodic" else "--",
                       label=f"{process}, b={b:g}")
    axes[1].set(xlabel="Trajectory length", ylabel="Variance of time averages",
                title="Finite-horizon dispersion")
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=.15)
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)
