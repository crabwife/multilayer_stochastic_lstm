"""Two-layer decoder with independently scheduled Beta forget gates."""

import math

import torch
from torch import nn
from torch.distributions import Beta


class LayeredRetentionLSTM(nn.Module):
    def __init__(self, width, policies):
        super().__init__()
        if len(policies) != 2 or any(p not in 'MSE' for p in policies):
            raise ValueError('Use two policies from M (mean), S (step), E (episode)')
        self.width, self.policies = width, policies
        self.encoder = nn.LSTM(1, width, batch_first=True)
        self.retention = nn.ModuleList(nn.Linear(width, 2 * width) for _ in policies)
        self.decoder = nn.ModuleList(nn.Linear(width + 1, 3 * width)
                                     for _ in policies)
        self.head = nn.Linear(width, 1)
        with torch.no_grad():
            for gate in self.retention:
                gate.bias[width:].fill_(2.5)

    def _initial(self, history, particles):
        _, (encoded, cell) = self.encoder(history)
        encoded, cell = encoded[0], cell[0]
        laws = []
        for gate in self.retention:
            raw_mean, raw_precision = gate(encoded).chunk(2, dim=-1)
            mean = torch.sigmoid(raw_mean).clamp(1e-4, 1 - 1e-4)
            precision = torch.nn.functional.softplus(raw_precision) + 2.
            laws.append((mean.repeat_interleave(particles, 0),
                         precision.repeat_interleave(particles, 0)))
        state = [(encoded.repeat_interleave(particles, 0),
                  cell.repeat_interleave(particles, 0))
                 for _ in self.policies]
        last = history[:, -1].repeat_interleave(particles, 0)
        retained = [self._draw(*law) if p == 'E' else None
                    for law, p in zip(laws, self.policies)]
        return state, laws, last, retained

    @staticmethod
    def _draw(mean, precision):
        return Beta((mean * precision).clamp_min(1e-4),
                    ((1 - mean) * precision).clamp_min(1e-4)).rsample()

    def _update(self, observed, last, state, laws, retained):
        incoming, next_state, next_retained = observed, [], []
        for layer, (policy, (h, c), (mean, precision), old) in enumerate(
            zip(self.policies, state, laws, retained)
        ):
            if policy == 'M':
                forget = mean
            elif policy == 'S':
                forget = self._draw(mean, precision)
            else:
                proposal = self._draw(mean, precision)
                forget = torch.where(observed.ne(last), proposal, old)
            ai, ao, ag = self.decoder[layer](torch.cat((incoming, h), -1)).chunk(3, -1)
            c = forget * c + torch.sigmoid(ai) * torch.tanh(ag)
            h = torch.sigmoid(ao) * torch.tanh(c)
            incoming = h
            next_state.append((h, c))
            next_retained.append(forget if policy == 'E' else None)
        return next_state, observed, next_retained

    def path_nll(self, history, targets, particles):
        particles = 1 if self.policies == 'MM' else particles
        state, laws, last, retained = self._initial(history, particles)
        targets = targets.repeat_interleave(particles, 0)
        logp = state[0][0].new_zeros(len(targets))
        for t in range(targets.shape[1]):
            logits = self.head(state[-1][0]).squeeze(-1)
            y = targets[:, t]
            logp -= torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction='none')
            if t + 1 < targets.shape[1]:
                state, last, retained = self._update(y[:, None], last, state,
                                                      laws, retained)
        logp = logp.reshape(len(history), particles)
        return -(torch.logsumexp(logp, 1) - math.log(particles)).mean()

    @torch.no_grad()
    def sample_paths(self, history, particles, horizon, emission_seed):
        state, laws, last, retained = self._initial(history, particles)
        generator = torch.Generator(device=history.device).manual_seed(emission_seed)
        paths = []
        for t in range(horizon):
            p = torch.sigmoid(self.head(state[-1][0]))
            y = (torch.rand(p.shape, device=p.device, dtype=p.dtype,
                            generator=generator) < p).to(p.dtype)
            paths.append(y.reshape(len(history), particles))
            if t + 1 < horizon:
                state, last, retained = self._update(y, last, state, laws, retained)
        return torch.stack(paths, -1)
