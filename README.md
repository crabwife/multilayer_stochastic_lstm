# Multilayer stochastic-memory LSTM | PDP

This is a Principled Data Processing project for testing where Beta gate stochasticity belongs in stacked LSTMs. A layer is **D** (deterministic), **F** (Beta forget), **I** (Beta input), or **A** (Beta forget/input/output); `AFD` orders layers bottom to top. The cell candidate stays deterministic. Every layer uses the same conditional gate-mean equations and Gaussian readout; each stochastic gate has a learned per-unit Beta precision. The model is implemented in `c_fit/src/model.py`, and every task is executed by its code-only notebook.

## Run

Create a Python 3.10+ environment, then:

```bash
pip install -r requirements.txt
make all
```

`make a_import`, `make b_prepare`, `make c_fit`, `make d_evaluate`, `make e_urn`, and `make f_plot` run the named task and all its predecessors. `make all` runs this original E001 pipeline. `make g_nonergodic` runs the independent synthetic benchmark without refitting E001; `make h_gate_ablation` reads its saved checkpoints without retraining. `make i_retention` runs the separate Beta retention study. `make j_layered_retention` compares Beta forget placement across two decoder layers and scores age-conditioned survival. The source notebooks are thin, code-only, and have one cell per activity; sibling `.py` files hold reusable functions. Executed notebooks and outputs go in the owning task's `output/`. A rerun reuses fitting checkpoints. For the complete E001 grid, change `model.grid` to `full` in `config.json` before the first run.

The import task uses the author-archived [DeSiervo et al. E001 data release](https://doi.org/10.5061/dryad.dbrv15f5t): first its Dryad CSV, then the authors' public GitHub repository if Dryad blocks the request. It does **not** call the EDI resource map, which can return HTTP 403. If both downloads fail, download `e001-aboveground-mass-2019-09-13.csv` from the DOI and place that single CSV in `a_import/input/`, then run `make all`. The exact source URL, SHA-256, and bytes are recorded in `a_import/output/manifest.json`. A prior raw cache from the EDI importer is rejected automatically. The archive has a one-line preamble; preparation detects and skips it and explicitly reads `mass.above` rather than the separate `mass` column. Review `b_prepare/output/audit.json` to verify columns, years, discarded rows, and plot coverage. If a downloaded CSV has different headers, set `data.columns` in `config.json`; rerun `make all`. Delete task outputs/checkpoints before changing the study years or source dataset.

## Task DAG

| Task | Input | Main output |
| --- | --- | --- |
| `a_import` | Archived Dryad/GitHub E001 or `a_import/input/*.csv` | Exact raw CSV, URL and SHA-256 manifest |
| `b_prepare` | Symlink to `a_import/output` | Audited plot-year panel, train-only scaler, annual windows |
| `c_fit` | Symlink to `b_prepare/output` | Screen, validation-selected finalists and checkpoints |
| `d_evaluate` | Symlinks to `b_prepare/output` and `c_fit/output` | Locked one-year and joint-path scores, paired contrasts, gate precision |
| `e_urn` | Traceable predecessor symlink | Trajectory-disjoint Pólya-urn falsification scores |
| `f_plot` | Symlinks to evaluation, urn, fit and preparation outputs | Seaborn figures, score table and report |
| `g_nonergodic` | Symlink to the shared gate-cell source in `c_fit/src` | Long-horizon binary process benchmark, exact process oracles and trajectory-level contrasts |
| `h_gate_ablation` | Symlinks to the benchmark output and shared sources | Paired gate-sampling intervention on saved checkpoints |
| `i_retention` | Symlink to the benchmark scoring source | Matched fresh versus episode-shared Beta forget draws; exact renewal survival and long-horizon scores |
| `j_layered_retention` | Symlink to retention data source | Two-layer forget schedule/order and age-conditioned episode survival |

All task folders contain `input/`, `output/`, and `src/`. Relative `input/` symlinks record upstream provenance. There are nine tasks, within the 26-letter PDP limit. The seed is **63** throughout; fitting replicates advance distinct reproducible portions of that stream. The default E001 screen covers all 1- and 2-layer strings plus 18 targeted 3-layer strings and reverse-order companions (38 total). The full grid contains all 84 depth-1 to depth-3 strings. Finalists include every homogeneous D/F/I/A stack and validation-selected stacks with their reverse orders, then five replicates. Test outcomes never determine finalists.

## What is tested

The **archived study subset** covers 1982–2004, before the later treatment changes, and pools E001 Fields A/B/C (54 plots per field in the study design). This is shorter than the current Cedar Creek E001 catalog, which lists 37 observed years through 2021. Each example uses six consecutive observed years of standardized `log1p` biomass to predict the next year. Missing years never become zeros. Target years through 1996 are training; 1997–2000 are validation; 2001–2004 are test. History may precede its target split, while all scaling and the high-biomass threshold are training-only. A three-year autoregressive forecast carries each particle's recurrent state and emission draw forward. Per-field, per-year scores and treatment metadata remain available for interpretation.

Training optimizes finite-particle predictive mixture NLL. The locked test evaluates NLL, CRPS, log-biomass MAE, 90% coverage, high-biomass Brier, joint path energy and variogram scores. Paired differences to the deterministic stack at the same depth resample **plots**, preserving all their annual windows; replicate differences are saved separately. The diagnostic files report Beta precision saturation, the gate contribution to predictive variance, and a within-history order permutation (last observation preserved). The Pólya urn has a random, path-dependent limiting fraction and uses whole-trajectory splits; it does not prove a property of the ecological dataset.

The comparison holds width and training budgets fixed and reports parameter counts. Stochastic variants add Beta precision parameters; a matched-parameter sensitivity and independent-site replication are needed for a general claim. A Gaussian biomass readout can also be misspecified. The repository contains the owner's E001 and short-urn run outputs; changes to configuration or data require fresh runs and checkpoint signatures enforce that.

## Long-horizon follow-up

Run `make g_nonergodic` for a separate, trajectory-disjoint benchmark with a 128-step future block. It compares the four gate modes across depths on a Pólya urn, an independent ergodic process, and a highly persistent but ergodic Markov chain. A Bernoulli readout matches the synthetic binary observations. Exact reference distributions let the score table show whether a model captures uncertainty in the future **time average**, not just next-step prediction. Read [the benchmark guide](g_nonergodic/README.md) for hypotheses, compute settings, outputs, and interpretation. This experiment does not infer nonergodicity from E001 biomass.

After that benchmark, `make h_gate_ablation` measures what sampled gates contribute **within each fitted model**. It reuses saved checkpoints and matches Bernoulli emission randomness between gate-on and gate-off rollouts. See [the ablation guide](h_gate_ablation/README.md).

## Beta retention timescale study

Run `make i_retention` for a controlled comparison of fresh versus episode-shared Beta forget coefficients at the same one-step gate distribution. Exact renewal survival formulas and 128–2048-step time-average simulations accompany a matched recurrent-forecast experiment on heavy-tailed and finite-mean controls. This task trains on 16-step joint paths and evaluates 32- and 128-step future frequencies. See [the retention study guide](i_retention/README.md) for the mechanism, hypotheses, outputs and limits. The model is a conventional recurrent experiment, not a reproduction of StoxLSTM.

## Layered retention placement

Run `make j_layered_retention` for matched two-layer Beta forget schedules `MM`, `MS`, `SM`, `ME`, `EM`, and `EE`. This task asks whether episode-shared draws in either layer predict the **conditional episode survival** law beyond a memoryless baseline. See [the layered retention guide](j_layered_retention/README.md) for scoring, controls and limits.
