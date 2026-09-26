"""Binary-emission counterpart of the shared four-mode GateLayer."""

import torch
from torch import nn
from model import GateLayer


class BinaryGateLSTM(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        if not config or any(mode not in "DFIA" for mode in config):
            raise ValueError(f"Unknown layer configuration {config!r}")
        self.width = width
        self.layers = nn.ModuleList([
            GateLayer(1 if j == 0 else width, width, mode)
            for j, mode in enumerate(config)
        ])
        self.head = nn.Linear(width, 1)

    def _history(self, x, particles):
        batch, length, _ = x.shape
        zseq = x.repeat_interleave(particles, dim=0)
        hs = [x.new_zeros((batch * particles, self.width)) for _ in self.layers]
        cs = [h.clone() for h in hs]
        for t in range(length):
            z = zseq[:, t, :]
            for j, layer in enumerate(self.layers):
                hs[j], cs[j] = layer(z, hs[j], cs[j])
                z = hs[j]
        return hs, cs

    def forward(self, x, particles=1):
        hs, _ = self._history(x, particles)
        return self.head(hs[-1]).reshape(len(x), particles).sigmoid().mean(dim=1)

    @torch.no_grad()
    def sample_paths(self, x, particles, horizon):
        batch = len(x)
        hs, cs = self._history(x, particles)
        draws = []
        for step in range(horizon):
            p = self.head(hs[-1]).sigmoid()
            y = torch.bernoulli(p)
            draws.append(y.reshape(batch, particles))
            if step + 1 < horizon:
                z = y
                for j, layer in enumerate(self.layers):
                    hs[j], cs[j] = layer(z, hs[j], cs[j])
                    z = hs[j]
        return torch.stack(draws, dim=-1)

    def precision_summary(self):
        values = []
        for layer_no, layer in enumerate(self.layers):
            for gate, raw in layer.precision.items():
                kappa = torch.nn.functional.softplus(raw) + 2
                values.append({"layer_bottom_zero": layer_no, "gate": gate,
                               "kappa_median": float(kappa.median().detach().cpu())})
        return values
