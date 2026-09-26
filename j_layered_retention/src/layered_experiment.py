"""Matched layer-order experiment with exact age-conditioned survival target."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import betaln
from torch.utils.data import DataLoader, TensorDataset

from retention_data import beta_shapes, generate, training_windows
from layered_model import LayeredRetentionLSTM


def oracle_survival(history, horizon, mean_stay, second_shape, process):
    """Exact next-horizon survival given a history starting at episode onset."""
    flip = history[:, 1:] != history[:, :-1]
    # Last change is at transition index k; episode age is L-2-k.
    age = np.where(flip.any(1), flip[:, ::-1].argmax(1), history.shape[1] - 1)
    if process == 'annealed':
        probability = np.full(len(history), mean_stay ** horizon)
    else:
        a, b = beta_shapes(mean_stay, second_shape)
        probability = np.exp(betaln(a + age + horizon, b) - betaln(a + age, b))
    return age, probability


def _signature(cfg, key, split):
    relevant = {'seed': cfg['seed'], 'retention': cfg['retention'],
                'layered_retention': cfg['layered_retention'], 'key': key}
    digest = hashlib.sha256(json.dumps(relevant, sort_keys=True).encode())
    for name in ('train', 'val'):
        digest.update(split[name].tobytes())
    return digest.hexdigest()


def _loaders(split, setting, seed):
    tensors = []
    for name in ('train', 'val'):
        x, y = training_windows(split[name], setting['context'],
                                setting['train_horizon'])
        tensors.append(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)))
    return (DataLoader(tensors[0], batch_size=setting['batch'], shuffle=True,
                       generator=torch.Generator().manual_seed(seed)),
            DataLoader(tensors[1], batch_size=setting['batch']))


def fit_one(cfg, key, split, config, rep, output):
    setting = cfg['retention'] | cfg['layered_retention']
    path = output / 'checkpoints' / key / f'{config}_rep{rep}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = _signature(cfg, key, split)
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError(f'Stale checkpoint {path}; clear j_layered_retention/output/checkpoints')
        return saved['summary']
    seed = cfg['seed'] + rep * 10_007
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LayeredRetentionLSTM(setting['width'], config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=setting['lr'],
                                  weight_decay=setting['weight_decay'])
    train, val = _loaders(split, setting, seed)
    best, best_epoch, state = float('inf'), 0, None
    started = time.time()
    for epoch in range(1, setting['epochs'] + 1):
        model.train()
        for x, y in train:
            optimizer.zero_grad(set_to_none=True)
            loss = model.path_nll(x.to(device), y.to(device), setting['particles_train'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        model.eval()
        total, n = 0., 0
        devices = list(range(torch.cuda.device_count())) if device.type == 'cuda' else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 301)
            for x, y in val:
                score = model.path_nll(x.to(device), y.to(device),
                                       setting['particles_val'])
                total += score.item() * len(y)
                n += len(y)
        score = total / n
        if score < best - 1e-4:
            best, best_epoch = score, epoch
            state = {name: value.detach().cpu().clone()
                     for name, value in model.state_dict().items()}
        if epoch - best_epoch >= setting['patience']:
            break
    summary = {'process': key, 'config': config, 'replicate': rep,
               'val_path_nll': best, 'epoch': best_epoch,
               'params': sum(p.numel() for p in model.parameters()),
               'seconds': time.time() - started}
    torch.save({'signature': signature, 'state': state, 'summary': summary}, path)
    return summary


def evaluate_one(cfg, key, split, config, rep, output):
    setting = cfg['retention'] | cfg['layered_retention']
    path = output / 'checkpoints' / key / f'{config}_rep{rep}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['signature'] != _signature(cfg, key, split):
        raise ValueError(f'Stale checkpoint {path}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LayeredRetentionLSTM(setting['width'], config).to(device)
    model.load_state_dict(saved['state'])
    model.eval()
    torch.manual_seed(cfg['seed'] + rep * 10_007 + 501)
    history = split['test'][:, :setting['context']]
    targets = split['test'][:, setting['context']:
                            setting['context'] + setting['test_horizon']]
    rows = []
    for start in range(0, len(history), setting['batch']):
        x = torch.from_numpy(history[start:start + setting['batch'], :, None]).to(device)
        paths = model.sample_paths(x, setting['particles_test'],
                                   setting['test_horizon'],
                                   cfg['seed'] + rep * 10_007 + 13_009 + start).cpu().numpy()
        for j, samples in enumerate(paths):
            index = start + j
            for horizon in setting['survival_horizons']:
                age, oracle = oracle_survival(history[index:index + 1], horizon,
                                               setting['mean_stay'], split['true_beta'],
                                               split['true_process'])
                observed = float(np.all(targets[index, :horizon] == history[index, -1]))
                survival = float(np.mean(np.all(samples[:, :horizon] == history[index, -1], axis=1)))
                memoryless = setting['mean_stay'] ** horizon
                rows.append({'process': key, 'config': config, 'replicate': rep,
                             'trajectory': index, 'horizon': horizon, 'age': int(age[0]),
                             'observed': observed, 'predicted': survival,
                             'oracle': float(oracle[0]),
                             'memoryless': memoryless,
                             'brier': (survival - observed) ** 2,
                             'oracle_brier': (float(oracle[0]) - observed) ** 2,
                             'memoryless_brier': (memoryless - observed) ** 2,
                             'excess_brier': (survival - observed) ** 2 -
                                             (float(oracle[0]) - observed) ** 2})
    return rows


def contrasts(scores, nboot, seed):
    rng = np.random.default_rng(seed + 1901)
    rows = []
    for (process, horizon), subset in scores.groupby(['process', 'horizon']):
        wide = subset.pivot(index=['trajectory', 'replicate'], columns='config',
                            values='excess_brier')
        for a, b in (('EM', 'ME'), ('EE', 'MM'), ('MS', 'SM'),
                     ('EM', 'MM'), ('ME', 'MM'), ('EE', 'EM')):
            differences = (wide[a] - wide[b]).groupby('trajectory').mean().to_numpy()
            draws = rng.choice(differences, (nboot, len(differences)),
                               replace=True).mean(1)
            rows.append({'process': process, 'horizon': horizon, 'contender': a,
                         'baseline': b, 'delta_excess_brier': float(differences.mean()),
                         'low95': float(np.quantile(draws, .025)),
                         'high95': float(np.quantile(draws, .975)),
                         'independent_trajectories': len(differences),
                         'training_replicates': int(subset.replicate.nunique())})
    return pd.DataFrame(rows)


def run(cfg, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    setting = cfg['retention'] | cfg['layered_retention']
    data = generate(cfg['retention'], cfg['seed'])
    (output / 'manifest.json').write_text(json.dumps({
        'seed': cfg['seed'], 'settings': setting,
        'selection': 'best 16-step validation joint path likelihood',
        'test': 'independent trajectories; exact conditional episode survival',
        'age': 'initial test history starts at the simulated episode onset',
        'inference': 'trajectory bootstrap conditions on dataset and training seeds'
    }, indent=2))
    validation, rows = [], []
    for key in setting['processes']:
        split = data[key]
        for config in setting['configs']:
            for rep in range(setting['repeats']):
                validation.append(fit_one(cfg, key, split, config, rep, output))
                rows.extend(evaluate_one(cfg, key, split, config, rep, output))
                pd.DataFrame(validation).to_csv(output / 'validation.csv', index=False)
                pd.DataFrame(rows).to_csv(output / 'trajectory_survival.csv', index=False)
    scores = pd.DataFrame(rows)
    scores['age_bin'] = pd.cut(scores.age, [-1, 3, 15, 31],
                               labels=['0-3', '4-15', '16-31'])
    scores.groupby(['process', 'config', 'horizon', 'age_bin'], observed=True,
                   as_index=False).agg(n=('observed', 'size'),
                                       observed=('observed', 'mean'),
                                       predicted=('predicted', 'mean'),
                                       oracle=('oracle', 'mean')).to_csv(
        output / 'age_reliability.csv', index=False)
    summary = scores.groupby(['process', 'config', 'horizon'], as_index=False).agg(
        brier=('brier', 'mean'), oracle_brier=('oracle_brier', 'mean'),
        memoryless_brier=('memoryless_brier', 'mean'),
        excess_brier=('excess_brier', 'mean'),
        predicted=('predicted', 'mean'), observed=('observed', 'mean'))
    summary.to_csv(output / 'summary.csv', index=False)
    contrasts(scores, setting['bootstrap'], cfg['seed']).to_csv(
        output / 'paired_contrasts.csv', index=False)
    return summary
