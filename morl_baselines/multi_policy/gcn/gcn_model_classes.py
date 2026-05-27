import torch as th
from torch import nn
from abc import ABC
import numpy as np

class BaseGCNModel(nn.Module, ABC):
    """Base Model for the GCN."""

    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int):
        """Initialize the GCN model."""
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.scaling_factor = nn.Parameter(th.tensor(scaling_factor).float(), requires_grad=False)
        self.hidden_dim = hidden_dim

    def forward(self, state, desired_return, desired_horizon):
        c = th.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c * self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        prediction = self.fc(s * c)
        return prediction

class DefaultGCNModel(BaseGCNModel):
    def __init__(self, state_dim: int, action_dim: int, reward_dim: int, scaling_factor: np.ndarray, hidden_dim: int, nr_layers: int = 1):
        """Initialize the GCN model."""
        super().__init__(state_dim, action_dim, reward_dim, scaling_factor, hidden_dim)
        self.nr_layers = nr_layers

        self.s_emb = nn.Sequential(nn.Linear(self.state_dim, self.hidden_dim), nn.Sigmoid())
        self.c_emb = nn.Sequential(nn.Linear(self.reward_dim + 1, self.hidden_dim), nn.Sigmoid())

        self.fc = nn.Sequential(
            *[
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ] * self.nr_layers,
            nn.Linear(self.hidden_dim, self.action_dim),
        )


