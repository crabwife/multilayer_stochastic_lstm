"""Context-conditioned Beta forget coefficient with controlled sampling schedule."""

import math

import torch
from torch import nn
from torch.distributions import Beta


class RetentionLSTM(nn.Module):
    """Same encoder, recurrent update, and parameters under all three policies.

    The forecast forget law is fixed given observed context. Only how often
    its Beta draw is refreshed changes; input/output gates remain deterministic.
    """

    def __init__(self, width, policy):
        super().__init__()
        if policy not in ("mean", "step", "episode"):
            raise ValueError(policy)
        self.width, self.policy = width, policy
        self.encoder = nn.LSTM(1, width, batch_first=True)
        self.retention = nn.Linear(width, 2 * width)
        with torch.no_grad():
            self.retention.bias[width:].fill_(2.5)
        self.decoder = nn.Linear(1 + width, 3 * width)
        self.head = nn.Linear(width, 1)

    def _initial(self, history, particles):
        _, (h, c) = self.encoder(history)
        h, c = h[0], c[0]
        raw_mean, raw_precision = self.retention(h).chunk(2, dim=-1)
        mean = torch.sigmoid(raw_mean).clamp(1e-4, 1 - 1e-4)
        precision = torch.nn.functional.softplus(raw_precision) + 2.
        h = h.repeat_interleave(particles, dim=0)
        c = c.repeat_interleave(particles, dim=0)
        mean = mean.repeat_interleave(particles, dim=0)
        precision = precision.repeat_interleave(particles, dim=0)
        last = history[:, -1, :].repeat_interleave(particles, dim=0)
        return h, c, mean, precision, last

    def _draw(self, mean, precision):
        return Beta((mean * precision).clamp_min(1e-4),
                    ((1 - mean) * precision).clamp_min(1e-4)).rsample()

    def _update(self, observed, last, h, c, mean, precision, retained):
        if self.policy == "mean":
            forget = mean
        elif self.policy == "step":
            forget = self._draw(mean, precision)
        else:
            # A new propensity begins with each observed/predicted state flip.
            proposal = self._draw(mean, precision)
            forget = torch.where(observed.ne(last), proposal, retained)
            retained = forget
        ai, ao, ag = self.decoder(torch.cat((observed, h), dim=-1)).chunk(3, dim=-1)
        c = forget * c + torch.sigmoid(ai) * torch.tanh(ag)
        h = torch.sigmoid(ao) * torch.tanh(c)
        return h, c, observed, retained

    def path_nll(self, history, targets, particles):
        """Monte Carlo marginal likelihood of the teacher-forced future path."""
        particles = 1 if self.policy == "mean" else particles
        h, c, mean, precision, last = self._initial(history, particles)
        retained = self._draw(mean, precision) if self.policy == "episode" else None
        targets = targets.repeat_interleave(particles, dim=0)
        total = h.new_zeros((len(h),))
        for step in range(targets.shape[1]):
            logit = self.head(h).squeeze(-1)
            y = targets[:, step]
            total = total - torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y, reduction="none")
            if step + 1 < targets.shape[1]:
                h, c, last, retained = self._update(
                    y[:, None], last, h, c, mean, precision, retained)
        total = total.reshape(len(history), particles)
        return -(torch.logsumexp(total, dim=1) - math.log(particles)).mean()

    @torch.no_grad()
    def sample_paths(self, history, particles, horizon, emission_seed):
        h, c, mean, precision, last = self._initial(history, particles)
        retained = self._draw(mean, precision) if self.policy == "episode" else None
        generator = torch.Generator(device=history.device).manual_seed(emission_seed)
        paths = []
        for step in range(horizon):
            probability = torch.sigmoid(self.head(h))
            y = (torch.rand(probability.shape, device=probability.device,
                            dtype=probability.dtype, generator=generator)
                 < probability).to(probability.dtype)
            paths.append(y.reshape(len(history), particles))
            if step + 1 < horizon:
                h, c, last, retained = self._update(
                    y, last, h, c, mean, precision, retained)
        return torch.stack(paths, dim=-1)

    @torch.no_grad()
    def retention_summary(self, history):
        _, (h, _) = self.encoder(history)
        raw_mean, raw_precision = self.retention(h[0]).chunk(2, dim=-1)
        mean = torch.sigmoid(raw_mean)
        precision = torch.nn.functional.softplus(raw_precision) + 2.
        return {"mean_forget": float(mean.mean().cpu()),
                "median_precision": float(precision.median().cpu()),
                "median_beta_second_shape": float(
                    (precision * (1 - mean)).median().cpu())}
