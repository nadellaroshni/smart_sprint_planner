"""
Experience Replay Buffer.

Stores (state, action_idx, reward, next_state, done) tuples.
Supports:
  - Uniform random sampling (standard DQN)
  - Prioritised sampling (PER) via TD-error magnitude

Both variants use a fixed-size circular buffer.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, NamedTuple, Optional

import numpy as np


class Transition(NamedTuple):
    state:      np.ndarray   # shape (FEATURE_DIM,)
    action_idx: int
    reward:     float
    next_state: np.ndarray   # shape (FEATURE_DIM,)
    done:       bool
    # Optional metadata (not used in sampling)
    task_id:    str = ""
    dev_id:     str = ""


class ReplayBuffer:
    """
    Circular experience replay buffer with optional priority weighting.

    Args:
        capacity:       Maximum number of transitions stored.
        alpha:          Priority exponent (0 = uniform, 1 = full priority).
        beta_start:     Initial IS-weight correction factor.
        beta_frames:    Frames over which beta anneals to 1.0.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.0,   # 0 = uniform (default); set > 0 for PER
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity    = capacity
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self._frame      = 1

        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._priorities: deque[float]  = deque(maxlen=capacity)
        self._max_priority: float       = 1.0

    # ── Public API ──────────────────────────────────────────────────────────

    def push(self, transition: Transition, td_error: Optional[float] = None) -> None:
        """Add a transition, optionally with an initial TD-error priority."""
        priority = (abs(td_error) + 1e-5) ** self.alpha if td_error is not None else self._max_priority
        self._buffer.append(transition)
        self._priorities.append(priority)
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> tuple:
        """
        Sample a batch of transitions.

        Returns:
            (transitions, indices, weights)
            - transitions: List[Transition]
            - indices:     List[int]  (for priority updates)
            - weights:     np.ndarray (importance-sampling weights, all 1.0 for uniform)
        """
        n = len(self._buffer)
        if n < batch_size:
            batch_size = n

        probs = np.array(self._priorities, dtype=np.float64)
        if self.alpha > 0:
            probs = probs ** self.alpha
            probs /= probs.sum()
        else:
            probs = np.ones(n) / n

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)

        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self._frame / self.beta_frames
        )
        self._frame += 1

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self._buffer[i] for i in indices]
        return transitions, indices.tolist(), weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        """Update priorities after a learning step (PER only)."""
        for i, err in zip(indices, td_errors):
            if i < len(self._priorities):
                p = (abs(err) + 1e-5) ** self.alpha
                self._priorities[i] = p
                self._max_priority = max(self._max_priority, p)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= 32   # minimum batch size