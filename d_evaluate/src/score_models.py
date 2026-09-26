import json
import numpy as np
import pandas as pd
import torch
from model import StackedGateLSTM
from fit_models import streams
from metrics import energy_score, gaussian_mixture_nll, paired_plot_bootstrap, sample_crps, variogram_score

def model_at(cfg, fit_output, name, replicate):
    state = torch.load(fit_output/"checkpoints"/"final"/f"{name}_rep{replicate}.pt", map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedGateLSTM(name, cfg["model"]["width"]).to(device)
    model.load_state_dict(state["state"])
    model.eval()
    return model

def one_step(model, x, batch, particles, seed):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    means, scales = [], []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            a, b = model(torch.as_tensor(x[start:start+batch], device=device), particles)
            means.append(a.cpu().numpy())
            scales.append(b.cpu().numpy())
    means, scales = np.concatenate(means), np.concatenate(scales)
    rng = np.random.default_rng(seed)
    draws = means + scales*rng.standard_normal(means.shape)
    return means, scales, draws

def paths(model, x, horizon, batch, particles, seed):
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    draws = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            samples = model.sample_paths(torch.as_tensor(x[start:start+batch], device=device), particles, horizon)
            draws.append(samples.cpu().numpy())
    return np.concatenate(draws)

def evaluate(cfg, arrays, metadata, selection, fit_output, prep_output, output):
    x, y = arrays["one_test_x"], arrays["one_test_y"][:, 0]
    px, py = arrays["path_test_x"], arrays["path_test_y"]
    one_meta, path_meta = metadata
    setting, seed = cfg["evaluation"], cfg["seed"]
    threshold = float(np.quantile(arrays["one_train_y"][:, 0], .9))
    scale = json.loads((prep_output/"scaler.json").read_text())["log1p_sd"]
    rng = np.random.default_rng(seed)
    shuffled = x.copy()
    for row in shuffled:
        row[:-1, 0] = rng.permutation(row[:-1, 0])
    one_rows, path_rows, gate_rows, order_rows = [], [], [], []
    for name in selection:
        for rep, draw_seed in enumerate(streams(cfg)):
            model = model_at(cfg, fit_output, name, rep)
            a, b, draws = one_step(model, x, cfg["fit"]["batch"], setting["particles"], draw_seed)
            lo, hi = np.quantile(draws, [.05, .95], axis=1)
            prob = (draws > threshold).mean(axis=1)
            row = one_meta.copy()
            row["config"], row["replicate"] = name, rep
            row["nll"] = gaussian_mixture_nll(a, b, y)
            row["crps"] = sample_crps(draws, y)
            row["mae_log"] = np.abs((a.mean(axis=1)-y)*scale)
            row["cover90"] = ((y >= lo) & (y <= hi)).astype(float)
            row["tail_brier"] = (prob-(y > threshold))**2
            row["gate_variance_share"] = a.var(axis=1)/(a.var(axis=1)+(b*b).mean(axis=1))
            one_rows.append(row)
            for item in model.precision_summary(): gate_rows.append({"config": name, "replicate": rep, **item})
            sa, sb, _ = one_step(model, shuffled, cfg["fit"]["batch"], setting["particles"], draw_seed)
            order_rows.append({"config": name, "replicate": rep,
                               "delta_nll_shuffled": float((gaussian_mixture_nll(sa, sb, y)-row.nll).mean())})
            sampled_paths = paths(model, px, py.shape[1], cfg["fit"]["batch"], setting["particles"], draw_seed)
            path_row = path_meta.copy()
            path_row["config"], path_row["replicate"] = name, rep
            path_row["energy"] = energy_score(sampled_paths, py)
            path_row["variogram"] = variogram_score(sampled_paths, py)
            path_rows.append(path_row)
    one, path = pd.concat(one_rows, ignore_index=True), pd.concat(path_rows, ignore_index=True)
    one.to_csv(output/"one_step.csv", index=False)
    path.to_csv(output/"paths.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output/"gate_precision.csv", index=False)
    pd.DataFrame(order_rows).to_csv(output/"order_sensitivity.csv", index=False)
    return one, path

def contrasts(cfg, one, path):
    rows = []
    for name in sorted(one.config.unique()):
        base = "D"*len(name)
        if name == base: continue
        for table, columns in [(one, ["nll", "crps", "mae_log", "tail_brier"]),
                               (path, ["energy", "variogram"])]:
            working = table.rename(columns={"replicate": "seed"})
            for metric in columns:
                rows.append(paired_plot_bootstrap(working, name, base, metric, cfg["evaluation"]["bootstrap"], cfg["seed"]))
    return pd.DataFrame(rows)
