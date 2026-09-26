"""Conditional-mean-matched deterministic and Beta gate cells."""

import math

import torch
from torch import nn
from torch.distributions import Beta


class GateLayer(nn.Module):
    def __init__(self, input_size, width, mode):
        super().__init__()
        if mode not in "DFIA":
            raise ValueError(f"Unknown layer mode {mode!r}")
        self.mode, self.width = mode, width
        self.affine = nn.Linear(input_size + width, 4 * width)
        active = {"D": "", "F": "f", "I": "i", "A": "fio"}[mode]
        self.precision = nn.ParameterDict({g: nn.Parameter(torch.full((width,), 2.5))
                                           for g in active})

    def _gate(self, logit, gate, sample):
        mean = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
        if gate not in self.precision or not sample:
            return mean
        k = (torch.nn.functional.softplus(self.precision[gate]) + 2).clamp(max=200)
        return Beta((k * mean).clamp_min(1e-4),
                    (k * (1 - mean)).clamp_min(1e-4)).rsample()

    def forward(self, x, h, c, sample=True):
        af, ai, ao, ag = self.affine(torch.cat((x, h), -1)).chunk(4, -1)
        f = self._gate(af, "f", sample)
        i = self._gate(ai, "i", sample)
        o = self._gate(ao, "o", sample)
        c = f * c + i * torch.tanh(ag)
        return o * torch.tanh(c), c


class StackedGateLSTM(nn.Module):
    def __init__(self, config, width):
        super().__init__()
        self.config = config
        self.width = width
        self.layers = nn.ModuleList([GateLayer(1 if j == 0 else width, width, mode)
                                     for j, mode in enumerate(config)])
        self.head = nn.Linear(width, 2)

    def _history(self, x, particles):
        b, t, _ = x.shape
        x = x.repeat_interleave(particles, dim=0)
        hs = [x.new_zeros((b * particles, self.width)) for _ in self.layers]
        cs = [v.clone() for v in hs]
        for step in range(t):
            z = x[:, step, :]
            for index, layer in enumerate(self.layers):
                hs[index], cs[index] = layer(z, hs[index], cs[index])
                z = hs[index]
        return hs, cs

    def _head(self, last, batch, particles):
        values = self.head(last).reshape(batch, particles, 2)
        return values[..., 0], torch.nn.functional.softplus(values[..., 1]) + 0.02

    def forward(self, x, particles=1):
        b = x.shape[0]
        hs, _ = self._history(x, particles)
        return self._head(hs[-1], b, particles)

    def sample_paths(self, x, particles, horizon):
        """One latent and emission trajectory per particle, with carried recurrent state."""
        batch = x.shape[0]
        hs, cs = self._history(x, particles)
        paths = []
        for step in range(horizon):
            mean, scale = self._head(hs[-1], batch, particles)
            y = mean + scale * torch.randn_like(mean)
            paths.append(y)
            if step + 1 < horizon:
                z = y.reshape(batch * particles, 1)
                for index, layer in enumerate(self.layers):
                    hs[index], cs[index] = layer(z, hs[index], cs[index])
                    z = hs[index]
        return torch.stack(paths, dim=-1)

    def precision_summary(self):
        result = []
        for layer_no, layer in enumerate(self.layers):
            for gate, raw in layer.precision.items():
                k = (torch.nn.functional.softplus(raw) + 2).clamp(max=200)
                result.append({"layer_bottom_zero": layer_no, "gate": gate,
                               "kappa_median": float(k.median().detach().cpu()),
                               "at_cap_fraction": float((k >= 199.9).float().mean().detach().cpu())})
        return result


def mixture_nll(mean, scale, target):
    logp = -.5 * (((target[:, None] - mean) / scale) ** 2
                  + 2 * torch.log(scale) + math.log(2 * math.pi))
    return -(torch.logsumexp(logp, dim=1) - math.log(mean.shape[1])).mean()
