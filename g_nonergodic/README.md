# Long-horizon nonergodicity benchmark

Run from the repository root after installing `requirements.txt`:

```bash
make g_nonergodic
```

This PDP task uses `input/fit` as a relative symlink to the existing gate-cell source in `c_fit/src`. Its source notebook is code-only; reusable code is in `src/`, and all generated results and executed notebooks go in `output/`. It does not download data or retrain E001. Change the `nonergodic` block of the root `config.json` to change the experiment. A smoke run can set `configs` to `["DD", "AA"]`, `processes` to `["polya"]`, `repeats` to 1, and `epochs` to 2; this only checks wiring and gives no reliable comparison. Delete `output/checkpoints/` when settings or data change; a signature check refuses stale checkpoints.

## Research question

Does stochasticity at different gates and depths preserve uncertainty about **future trajectory averages** beyond what a deterministic recurrent model with a Bernoulli head can represent? `D`, `F`, `I`, and `A` use the same gate equations and width as the E001 models; only the emission changes from Gaussian to Bernoulli to fit the binary experiments. `FI` means stochastic forget below stochastic input. The default grid includes one, two, and three layers, with two training replicates each. Models have slightly different parameter counts because stochastic gates learn precision.

| Process | Mechanism | Correct future-count law from observed context |
| --- | --- | --- |
| `polya` | Two-color Pólya urn initialized with one of each color | Beta-binomial with `alpha=1+ones`, `beta=1+zeros` |
| `iid` | Ergodic Bernoulli(0.5) | Binomial with `p=0.5` |
| `sticky` | Ergodic two-state Markov chain with known stay probability | Exact count distribution computed by dynamic programming |

The urn's binary sequence is **observationally identical** to a latent `p ~ Beta(1,1)` drawn once per trajectory followed by iid Bernoulli(`p`) outcomes. No prediction score here can tell reinforcement from that latent-mixture explanation. The sticky control has long correlations but converges to a fixed stationary mean, so a broad finite-horizon prediction alone does not establish nonergodicity.

Trajectories are split before windows are constructed. Training and validation use three disjoint 32-observation contexts per trajectory and one-step Bernoulli log loss. The test uses the first 32 observations, a **128-step** held-out future, and 64 fully autoregressive samples per model; no test outcomes select configurations. The oracle conditions on exactly the same observed context. No long-run limit is observed directly: these are finite future-block frequencies.

## Read the outputs

- `manifest.json`: seeds, mechanisms, split and scoring settings.
- `validation.csv`: one-step validation selection, training time and parameter counts.
- `trajectory_scores.csv`: one row per test trajectory, configuration and replicate. Contains frequency CRPS (lower is better), one-step log loss, 90% interval coverage, predicted frequency variance, and corresponding exact-oracle values.
- `summary.csv`: mean scores per process/configuration; the oracle columns give a finite-sample reference, not a training baseline.
- `paired_contrasts.csv`: difference versus `D`, `DD`, or `DDD` at the same depth, resampling **independent test trajectories** while keeping replicate pairs together. Multiple exploratory comparisons are unadjusted.
- `gate_precision.csv`: learned per-layer Beta precision. Precision near its initialization plus poor long-horizon dispersion is evidence the gate noise contributed little under this objective; it does not prove the architecture cannot learn another task.

The first useful readout is whether the variance and CRPS of the predicted 128-step frequency match the urn reference while staying sharp on iid and sticky data. A gain on one-step loss without improved frequency CRPS does not support the long-horizon hypothesis. Because training uses a one-step objective and independent gate draws at each step, the benchmark is deliberately capable of exposing a mismatch; a persistent trajectory-level latent model is a strong future comparator. Compute cost scales with configurations × processes × replicates, then with sampled trajectories × particles × horizon.
