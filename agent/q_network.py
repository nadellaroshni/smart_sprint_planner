"""
Q-Network — pure NumPy implementation.

Why NumPy and not PyTorch?
  - No GPU dependency, runs on CPU within 8 GB RAM
  - Portable: works in any Python env without CUDA
  - Sufficient for this problem size (action space ≤ 48)

Architecture:
  Input  → Linear(155) → ReLU → Linear(128) → ReLU → Linear(64) → Linear(1)

The network is used as a Q-value approximator for a fixed action set.
For each candidate (task_id, dev_id) pair we concatenate the state vector
with a one-hot action embedding, then predict Q(s, a).

Alternative: output all Q-values at once if action space is small and fixed.
We use the concatenation approach so the network generalises to variable
action spaces without architectural changes.

Training uses:
  - MSE loss
  - Adam optimiser (implemented from scratch, ~40 lines)
  - Gradient clipping (norm ≤ 1.0)
  - Double-DQN target network (updated every N steps)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .features import FEATURE_DIM, MAX_TASKS, MAX_DEVS

logger = logging.getLogger(__name__)

# ── Action embedding size ────────────────────────────────────────────────────
# We encode (task_idx, dev_idx) as a 2-value normalised pair
ACTION_DIM = 2
INPUT_DIM  = FEATURE_DIM + ACTION_DIM     # 157 + 2 = 159


# ── Activation helpers ────────────────────────────────────────────────────────

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float32)


# ── Adam optimiser state ──────────────────────────────────────────────────────

class AdamState:
    def __init__(self, shape, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(shape, dtype=np.float32)
        self.v     = np.zeros(shape, dtype=np.float32)
        self.t     = 0

    def step(self, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        m_hat  = self.m / (1 - self.beta1 ** self.t)
        v_hat  = self.v / (1 - self.beta2 ** self.t)
        return self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Q-Network ─────────────────────────────────────────────────────────────────

class QNetwork:
    """
    3-hidden-layer MLP: INPUT_DIM → 128 → 64 → 32 → 1

    Parameters are stored as plain numpy arrays.
    Supports forward pass, MSE loss, backprop, Adam updates.
    """

    def __init__(self, lr: float = 1e-3, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.lr = lr

        # He initialisation
        def he(fan_in, fan_out):
            return rng.standard_normal((fan_in, fan_out)).astype(np.float32) * np.sqrt(2.0 / fan_in)

        # Weights
        self.W1 = he(INPUT_DIM, 128)
        self.b1 = np.zeros((1, 128), dtype=np.float32)
        self.W2 = he(128, 64)
        self.b2 = np.zeros((1, 64), dtype=np.float32)
        self.W3 = he(64, 32)
        self.b3 = np.zeros((1, 32), dtype=np.float32)
        self.W4 = he(32, 1)
        self.b4 = np.zeros((1, 1), dtype=np.float32)

        params = [self.W1, self.b1, self.W2, self.b2,
                  self.W3, self.b3, self.W4, self.b4]
        self._opt = [AdamState(p.shape, lr=lr) for p in params]
        self._params = params

        # Cache for backprop
        self._cache: dict = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (batch, INPUT_DIM)
        returns: (batch, 1) Q-values
        """
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = relu(z3)
        q  = a3 @ self.W4 + self.b4

        self._cache = {"x": x, "z1": z1, "a1": a1,
                       "z2": z2, "a2": a2, "z3": z3, "a3": a3}
        return q

    def predict(self, x: np.ndarray) -> float:
        """Single-sample prediction (no grad)."""
        return float(self.forward(x.reshape(1, -1))[0, 0])

    def update(self, x: np.ndarray, target: np.ndarray, weight: float = 1.0) -> float:
        """
        One gradient step.
        x:      (batch, INPUT_DIM)
        target: (batch, 1)
        Returns MSE loss.
        """
        q = self.forward(x)
        err   = q - target
        loss  = float(np.mean(err ** 2) * weight)

        # Backprop
        dq   = 2 * err / len(x) * weight
        da3  = dq @ self.W4.T
        dW4  = self._cache["a3"].T @ dq
        db4  = dq.sum(axis=0, keepdims=True)

        dz3  = da3 * relu_grad(self._cache["z3"])
        da2  = dz3 @ self.W3.T
        dW3  = self._cache["a2"].T @ dz3
        db3  = dz3.sum(axis=0, keepdims=True)

        dz2  = da2 * relu_grad(self._cache["z2"])
        da1  = dz2 @ self.W2.T
        dW2  = self._cache["a1"].T @ dz2
        db2  = dz2.sum(axis=0, keepdims=True)

        dz1  = da1 * relu_grad(self._cache["z1"])
        dW1  = self._cache["x"].T @ dz1
        db1  = dz1.sum(axis=0, keepdims=True)

        grads = [dW1, db1, dW2, db2, dW3, db3, dW4, db4]

        # Gradient clipping (global norm)
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
        if global_norm > 1.0:
            grads = [g / global_norm for g in grads]

        # Adam updates
        for param, grad, opt in zip(self._params, grads, self._opt):
            param -= opt.step(grad)

        return loss

    def copy_weights_from(self, other: "QNetwork") -> None:
        """Copy weights from another network (for target network sync)."""
        for sp, op in zip(self._params, other._params):
            sp[:] = op[:]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._params, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            saved = pickle.load(f)
        for sp, sv in zip(self._params, saved):
            sp[:] = sv[:]
        logger.info(f"Model loaded from {path}")
