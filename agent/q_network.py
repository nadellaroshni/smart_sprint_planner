"""
Q-network implemented with PyTorch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .features import ACTION_FEATURE_DIM, FEATURE_DIM

logger = logging.getLogger(__name__)

ACTION_DIM = ACTION_FEATURE_DIM
INPUT_DIM = FEATURE_DIM + ACTION_DIM


class QNetwork:
    """
    Multi-layer perceptron: INPUT_DIM -> 192 -> 128 -> 64 -> 1
    """

    def __init__(self, lr: float = 1e-3, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = nn.Sequential(
            nn.Linear(INPUT_DIM, 192),
            nn.ReLU(),
            nn.LayerNorm(192),
            nn.Linear(192, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def forward(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            return self.model(tensor).detach().cpu().numpy()

    def predict(self, x: np.ndarray) -> float:
        with torch.no_grad():
            tensor = torch.as_tensor(x.reshape(1, -1), dtype=torch.float32, device=self.device)
            return float(self.model(tensor).item())

    def update(self, x: np.ndarray, target: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> float:
        self.model.train()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        target_tensor = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if sample_weights is None:
            weight_tensor = torch.ones((len(x), 1), dtype=torch.float32, device=self.device)
        else:
            weight_tensor = torch.as_tensor(sample_weights, dtype=torch.float32, device=self.device).reshape(-1, 1)

        pred = self.model(x_tensor)
        loss = (self.loss_fn(pred, target_tensor) * weight_tensor).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.item())

    def copy_weights_from(self, other: "QNetwork") -> None:
        self.model.load_state_dict(other.model.state_dict())

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        logger.info("Model loaded from %s", path)
