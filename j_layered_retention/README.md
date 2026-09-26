# Layer placement and episode survival

Run `make j_layered_retention` from the repository root in the `research313` environment. This independent PDP task writes checkpoints and CSV outputs under `j_layered_retention/output/`. It uses the shared renewal simulator through a relative link in `input/`, without rerunning the earlier pipeline. The notebook has one code cell per activity; `src/layered_model.py` contains the model.

## Question

Does holding a Beta forget draw for an episode help a two-layer recurrent decoder learn the **age-dependent probability of no state flip**? Does it matter which layer receives the persistent draw? Each configuration has exactly the same encoder, two decoder layers, gates, readout and parameter count. Only the forget-gate refresh schedules differ. The first letter is the lower layer, the second the upper layer:

| Code | Lower layer | Upper layer |
| --- | --- | --- |
| `MM` | conditional Beta mean | conditional Beta mean |
| `MS`, `SM` | mean, fresh draw each step | fresh draw each step, mean |
| `ME`, `EM` | mean, episode-shared draw | episode-shared draw, mean |
| `EE` | episode-shared draw | episode-shared draw |

`E` samples a gate coefficient at the start of the forecast and resamples it when the **observed or generated output state flips**. Each layer has its own context-conditioned Beta law. The decoder uses ordinary deterministic input/output/candidate gates and Bernoulli outputs. This is a controlled Beta-forget LSTM-style decoder, not a reproduction of StoxLSTM.

Three locked datasets are used: annealed `b=0.6` (short-memory control), episodic `b=0.6` (infinite mean episode duration) and episodic `b=1.4` (finite mean). They share one-step stay probability `0.95`. Training uses disjoint trajectories and fixed 16-step joint-path likelihood; validation selects epochs without looking at test trajectories. Two training seeds per configuration are the default. The 128-step autoregressive test reports survival after 8, 32 and 128 transitions. The task fits 36 models with default settings; it may take substantial time on CPU. Checkpoints resume a completed fit and reject changed input/settings. To do a quick wiring run, edit only `layered_retention.repeats` and `layered_retention.epochs` before the first run; clear **this task's** checkpoints when restoring the settings.

## Direct score

For the episodic process, an episode of observed age `A` has posterior stay propensity `Beta(a+A,b)`, and its exact probability of surviving another `T` transitions is `B(a+A+T,b)/B(a+A,b)`. For the annealed process it is `0.95**T`, regardless of `A`. Here the test context begins at simulation time zero, when an episode begins, so `A` is observed without left censoring. `trajectory_survival.csv` records each held-out event, ensemble survival forecast, oracle forecast, and Brier scores. `summary.csv` includes the exact-oracle and memoryless Brier benchmarks; excess Brier subtracts the oracle's score on the *same events*. `age_reliability.csv` checks age bins. `paired_contrasts.csv` averages seed-wise score differences per independent trajectory and bootstraps trajectories, with negative differences favoring the contender. The prespecified order comparison is `EM` versus `ME`.

A promising result would be an `EM`/`ME`/`EE` gain on episodic `b=0.6` that persists across 8 and 32 steps, shows age calibration, and is absent on annealed control. A gain only at 128 steps in predicted dispersion does not establish learning the episode law. The test is finite-horizon; `b=1.4` can also look persistent. Two seeds and one simulated dataset cannot support a general or novelty claim. The exact survival formula describes the **data process**; learned gate coefficients are not identifiable as its latent stay propensity. Follow up any signal with independent datasets, more seeds, larger horizons and a standard gated recurrent baseline.
