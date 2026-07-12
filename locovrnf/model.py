"""Affordance density model (reused from Layout-NF): p(visit offset | patch).

A small CNN encodes the 48x48 ego-centric occupancy crop into a context
vector; a zuko conditional Neural Spline Flow models the 2D visit offset.
Identical skeleton to layoutnf.affordance so results are comparable across the
synthetic (TRUMANS) and real (LocoReal/LocoVR) datasets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import zuko


class CropCNN(nn.Module):
    """48x48 binary occupancy crop -> feature vector."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),   # 24x24
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 12x12
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),  # 6x6
            nn.Flatten(), nn.Linear(32 * 36, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, crop):          # [B,48,48] float
        return self.net(crop[:, None])


class AffordanceFlow(nn.Module):
    def __init__(self, ctx: int = 64):
        super().__init__()
        self.cnn = CropCNN(ctx)
        self.flow = zuko.flows.NSF(features=2, context=ctx,
                                   transforms=4, hidden_features=(128, 128))

    def log_prob(self, crop, x):
        return self.flow(self.cnn(crop)).log_prob(x)


class StepFlow(nn.Module):
    """Autoregressive goal-conditioned step density (C3).

    p( next-step ego displacement (dfwd, dlat) | occupancy patch ahead,
       goal vector in ego frame, current speed ).
    Rolling this out start->goal yields a *distribution* over routes, which is
    the re-design simulator: swap the occupancy patch (move furniture) and the
    sampled routes re-plan around it.
    """

    def __init__(self, crop_dim: int = 48, extra: int = 4, ctx: int = 64):
        super().__init__()
        self.cnn = CropCNN(crop_dim)
        self.mlp = nn.Sequential(nn.Linear(extra, 32), nn.ReLU(),
                                 nn.Linear(32, 32), nn.ReLU())
        self.merge = nn.Sequential(nn.Linear(crop_dim + 32, ctx), nn.ReLU())
        self.flow = zuko.flows.NSF(features=2, context=ctx,
                                   transforms=5, hidden_features=(128, 128))

    def context(self, crop, extra):
        return self.merge(torch.cat([self.cnn(crop), self.mlp(extra)], -1))

    def log_prob(self, crop, extra, step):
        return self.flow(self.context(crop, extra)).log_prob(step)

    def sample(self, crop, extra, n: int = 1):
        return self.flow(self.context(crop, extra)).sample((n,))
