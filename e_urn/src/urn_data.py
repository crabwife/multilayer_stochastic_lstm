import numpy as np
import pandas as pd
import torch
from prepare_data import windows
from fit_models import fit_one, streams
from model import StackedGateLSTM, mixture_nll
from metrics import energy_score, variogram_score

def generate(cfg):
    setting = cfg["urn"]
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for plot in range(setting["trajectories"]):
        red, total = 1, 2
        for year in range(setting["years"]):
            rows.append((f"urn:{plot:04}", year, red/total, "none"))
            red += int(rng.random() < red/total)
            total += 1
    panel = pd.DataFrame(rows, columns=["plot_id", "year", "z", "treatment"])
    plots = np.array(sorted(panel.plot_id.unique()))
    rng.shuffle(plots)
    groups = [set(x.tolist()) for x in np.split(plots, [int(.7*len(plots)), int(.85*len(plots))])]
    arrays = {}
    for prefix, horizon in [("one", 1), ("path", cfg["evaluation"]["horizon"])]:
        x, y, meta = windows(panel, cfg["split"]["lookback"], horizon)
        for label, members in zip(["train", "val", "test"], groups):
            mask = meta.plot_id.isin(members).to_numpy()
            arrays[f"{prefix}_{label}_x"] = x[mask]
            arrays[f"{prefix}_{label}_y"] = y[mask]
    return arrays

def run(cfg, arrays, output):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in cfg["urn"]["configs"]:
        for rep, seed in enumerate(streams(cfg)):
            fit = fit_one(cfg, arrays, name, rep, "urn", output, cfg["urn"]["epochs"])
            checkpoint = torch.load(output/"checkpoints"/"urn"/f"{name}_rep{rep}.pt", map_location="cpu", weights_only=True)
            model = StackedGateLSTM(name, cfg["model"]["width"]).to(device)
            model.load_state_dict(checkpoint["state"])
            model.eval()
            torch.manual_seed(seed)
            x, y = arrays["one_test_x"], arrays["one_test_y"][:, 0]
            total = 0.
            with torch.no_grad():
                for start in range(0, len(x), cfg["fit"]["batch"]):
                    xx = torch.as_tensor(x[start:start+cfg["fit"]["batch"]], device=device)
                    yy = torch.as_tensor(y[start:start+cfg["fit"]["batch"]], device=device)
                    a, b = model(xx, cfg["evaluation"]["particles"])
                    total += mixture_nll(a, b, yy).item()*len(yy)
                path_x, path_y = arrays["path_test_x"], arrays["path_test_y"]
                sampled = []
                for start in range(0, len(path_x), cfg["fit"]["batch"]):
                    xx = torch.as_tensor(path_x[start:start+cfg["fit"]["batch"]], device=device)
                    sampled.append(model.sample_paths(xx, cfg["evaluation"]["particles"], path_y.shape[1]).cpu().numpy())
            sampled = np.concatenate(sampled)
            rows.append({**fit, "test_nll": total/len(x),
                         "path_energy": float(energy_score(sampled, path_y).mean()),
                         "path_variogram": float(variogram_score(sampled, path_y).mean())})
    return pd.DataFrame(rows)
