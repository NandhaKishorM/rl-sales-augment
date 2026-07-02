"""Inference-side model classes for the trained RL sales policy.

`MultiActorCritic` is the network architecture the bundled policy weights load into
(factored policy: one categorical head per deal); `ScratchFeaturizer` turns the
world observation into the policy's input features. Only the forward pass ships --
the PPO training pipeline, the GPU-vectorized world, and the company fine-tuning
stack are the proprietary training assets of Convai Innovations Pvt. Ltd.
(contact nandakishor@convaiinnovations.com).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ScratchFeaturizer:
    input_mode = "numeric"
    def __init__(self, obs_dim, device="cpu"):
        self.feature_dim = obs_dim; self.device = device
    @torch.no_grad()
    def featurize(self, obs, texts):
        return torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)


class MultiActorCritic(nn.Module):
    def __init__(self, feature_dim, hidden, n_settlements, n_actions):
        super().__init__()
        self.n, self.a = n_settlements, n_actions
        self.body = nn.Sequential(nn.Linear(feature_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, n_settlements * n_actions)
        self.critic = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2)); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1.0)

    def forward(self, feats):
        """Deterministic head: (logits (B,n,a), value (B,))."""
        h = self.body(feats)
        return self.actor(h).view(-1, self.n, self.a), self.critic(h).squeeze(-1)
