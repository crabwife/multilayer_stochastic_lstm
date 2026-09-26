# Checkpoint-only gate intervention

After running `make g_nonergodic`, run `make h_gate_ablation` from the repository root. This task reads the saved `g_nonergodic/output/checkpoints/`; it **does not retrain** the 72 models. Keep the original `config.json` and checkpoints together: signatures reject a mismatched setting or dataset. The task's `input/` symlinks point to the benchmark output, its source and the shared gate source. The code-only notebook orchestrates the analysis; results go in `h_gate_ablation/output/`.

The intervention uses each trained model twice on the **same test trajectories** and weights:

- **On:** draw each active gate from its learned Beta distribution at every step.
- **Off:** replace every active gate draw by its conditional mean. Recurrent states and emissions still evolve normally.

A separate, fixed Bernoulli random stream supplies the same emission uniforms in the paired runs. Thus an observed difference in future-frequency CRPS arises from the gate intervention and its downstream effects, rather than a shifted emission random stream. `D`, `DD` and `DDD` must yield exactly zero differences; the notebook aborts otherwise. Both runs use 64 path particles, the same 128-step horizon and the same 48 held-out trajectories per process as the original benchmark. They use the original train/validation split only to verify checkpoint signatures.

`summary.csv` reports each policy and the **on minus off** differences for CRPS and future-frequency variance, alongside exact oracle values. Lower CRPS is better, so a negative `delta_crps` favors sampled gates. `paired_contrasts.csv` gives trajectory-bootstrap intervals, averaging the two training replicates within each trajectory. `trajectory_scores.csv` keeps the paired per-trajectory results; `manifest.json` records the intervention. The intervals condition on the fitted checkpoints and do not quantify uncertainty across new training seeds or new simulated datasets. Comparing many layouts remains exploratory.

This is a causal **inference-time intervention on a fitted model**, not a claim that setting gates to their means is the best deterministic model. If gates-on helps only on the sticky ergodic chain, the advantage is not specific to nonergodicity. If it helps the Pólya urn but both policies remain well behind the exact oracle, the architecture may still lack a persistent trajectory-level latent variable. The original benchmark's empirical 64-particle CRPS has a small finite-ensemble upward bias relative to the exact oracle's CRPS; the paired on/off comparison shares the same particle count and emission stream.
