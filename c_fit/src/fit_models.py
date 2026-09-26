import itertools
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import StackedGateLSTM, mixture_nll

def grid(cfg):
    depth = cfg["model"]["max_depth"]
    if cfg["model"]["grid"] == "full":
        return ["".join(v) for n in range(1, depth+1) for v in itertools.product("DFIA", repeat=n)]
    short = ["".join(v) for n in range(1, min(depth, 2)+1) for v in itertools.product("DFIA", repeat=n)]
    deep = ["DDD", "FFF", "III", "AAA", "AFD", "ADF", "FAD", "FDA", "IAD", "AID", "FAI", "AIF"] if depth >= 3 else []
    return short + list(dict.fromkeys(deep + [x[::-1] for x in deep]))

def streams(cfg):
    return [cfg["seed"]]*cfg["fit"]["repeats"]

def batches(x, y, batch, shuffle, seed, replicate):
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y[:, 0], dtype=torch.float32))
    generator = torch.Generator().manual_seed(seed)
    for _ in range(replicate): torch.randperm(len(dataset), generator=generator)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, generator=generator)

def fit_one(cfg, arrays, config, replicate, tag, output, epochs=None):
    from pathlib import Path
    output = Path(output)
    seed = streams(cfg)[replicate]
    width, settings = cfg["model"]["width"], cfg["fit"]
    file = output/"checkpoints"/tag/f"{config}_rep{replicate}.pt"
    file.parent.mkdir(parents=True, exist_ok=True)
    # Keep pre-benchmark checkpoint identities stable: the independent
    # nonergodic task cannot affect E001 or short-urn training data.
    legacy_cfg = {key: value for key, value in cfg.items()
                  if key not in ("nonergodic", "retention", "layered_retention")}
    signature = hashlib.sha256(json.dumps(legacy_cfg, sort_keys=True).encode()
        + arrays["one_train_x"].tobytes() + arrays["one_train_y"].tobytes()
        + arrays["one_val_x"].tobytes() + arrays["one_val_y"].tobytes()).hexdigest()
    if file.exists():
        checkpoint = torch.load(file, map_location="cpu", weights_only=True)
        if checkpoint["signature"] != signature:
            raise ValueError(f"Checkpoint input/config changed: {file}; clear c_fit/output/checkpoints")
        return checkpoint["summary"]
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if replicate: torch.rand(1024*replicate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedGateLSTM(config, width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    train = batches(arrays["one_train_x"], arrays["one_train_y"], settings["batch"], True, seed, replicate)
    val = batches(arrays["one_val_x"], arrays["one_val_y"], settings["batch"], False, seed, replicate)
    best, best_epoch, best_state = float("inf"), 0, None
    started = time.time()
    for epoch in range(1, (epochs or settings[f"{tag}_epochs"])+1):
        model.train()
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            a, b = model(x.to(device), settings["train_particles"])
            loss = mixture_nll(a, b, y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        score, n = 0., 0
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for x, y in val:
                a, b = model(x.to(device), settings["val_particles"])
                score += mixture_nll(a, b, y.to(device)).item()*len(y)
                n += len(y)
        score /= n
        if score < best-1e-4:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch-best_epoch >= settings["patience"]: break
    summary = {"config": config, "replicate": replicate, "val_nll": best, "epoch": best_epoch,
               "params": sum(p.numel() for p in model.parameters()), "seconds": time.time()-started}
    torch.save({"state": best_state, "summary": summary, "signature": signature}, file)
    return summary

def finalists(cfg, screen):
    ranked = screen.groupby("config", as_index=False).val_nll.mean()
    depth, top = cfg["model"]["max_depth"], cfg["fit"]["top_per_depth"]
    keep = {letter*n for n in range(1, depth+1) for letter in "DFIA"}
    for n in range(1, depth+1):
        names = ranked[ranked.config.str.len() == n].nsmallest(top, "val_nll").config
        keep.update(names)
        keep.update(x[::-1] for x in names)
    return sorted(keep, key=lambda x: (len(x), x))
