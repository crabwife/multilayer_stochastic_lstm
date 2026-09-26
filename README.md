# Multilayer stochastic-memory LSTM

This is a PDP organized project for testing where Beta gate stochasticity belongs in stacked LSTMs. A layer is **D** (deterministic), **F** (Beta-forget), **I** (Beta-input), or **A** (Beta-forget/input/output); `AFD` orders layers bottom to top. The cell candidate stays deterministic. Every layer uses the same conditional gate-mean equations and Gaussian readout; each stochastic gate has a learned per-unit Beta precision. The model is implemented in `c_fit/src/model.py`, and every task is executed by its code-only notebook.


The import task uses the author-archived [DeSiervo et al. E001 data release](https://doi.org/10.5061/dryad.dbrv15f5t): first its Dryad CSV, then the authors' public GitHub repository if Dryad blocks the request. It does **not** call the EDI resource map, which can return HTTP 403.

## What is tested

Training optimizes finite-particle predictive mixture NLL. The locked test evaluates NLL, CRPS, log-biomass MAE, 90% coverage, high-biomass Brier, joint path energy and variogram scores. Paired differences to the deterministic stack at the same depth resample **plots**, preserving all their annual windows; replicate differences are saved separately. The diagnostic files report Beta precision saturation, the gate contribution to predictive variance, and a within-history order permutation (last observation preserved). The Pólya urn has a random, path-dependent limiting fraction and uses whole-trajectory splits; it does not prove a property of the ecological dataset.

