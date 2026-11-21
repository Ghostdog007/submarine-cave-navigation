"""Feature extractors for dict observations (camera image + state vector).

CameraStateExtractor:
    camera (1, 64) --Conv1d(5, s2)-ReLU-Conv1d(5, s2)-ReLU-Conv1d(3, s2)-ReLU--> flatten --Linear--> 64
    state  (k,)    --Linear-ReLU--> 64
    concatenated (128) -> PPO's actor and critic heads (net_arch).
The 1D convolutions look for local patterns along the scanline (wall edges,
gaps, the brightness gradient of a nearby wall); stride 2 halves the width
each layer: 64 -> 32 -> 16 -> 8 positions.
"""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class CameraStateExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, channels=(16, 32, 32),
                 camera_features: int = 64, state_features: int = 64):
        cam_shape = observation_space["camera"].shape          # (C, L)
        state_dim = observation_space["state"].shape[0]
        super().__init__(observation_space, features_dim=camera_features + state_features)
        c1, c2, c3 = channels
        self.cnn = nn.Sequential(
            nn.Conv1d(cam_shape[0], c1, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.cnn(torch.zeros(1, *cam_shape)).shape[1]
        self.camera_fc = nn.Sequential(nn.Linear(n_flat, camera_features), nn.ReLU())
        self.state_fc = nn.Sequential(nn.Linear(state_dim, state_features), nn.ReLU())

    def forward(self, obs: dict) -> torch.Tensor:
        return torch.cat([self.camera_fc(self.cnn(obs["camera"])), self.state_fc(obs["state"])], dim=1)


EXTRACTORS = {"camera_state": CameraStateExtractor}
