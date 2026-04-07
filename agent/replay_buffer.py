"""
Experience replay buffer for DDQN.
"""

from __future__ import annotations

from collections import deque
from typing import List, NamedTuple, Optional

import numpy as np


class Transition(NamedTuple):
    state: np.ndarray
    action_features: np.ndarray
    reward: float
    next_state: np.ndarray
    next_valid_action_features: List[np.ndarray]
    done: bool
    task_id: str = ""
    dev_id: str = ""


class ReplayBuffer:
    def __init__(
        self,
        capacity: int = 10_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self._frame = 1
        self._buffer: deque[Transition] = deque(maxlen=capacity)
        self._priorities: deque[float] = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(self, transition: Transition, td_error: Optional[float] = None) -> None:
        priority = (abs(td_error) + 1e-5) if td_error is not None else self._max_priority
        self._buffer.append(transition)
        self._priorities.append(priority)
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> tuple:
        n = len(self._buffer)
        if n < batch_size:
            batch_size = n

        probs = np.array(self._priorities, dtype=np.float64)
        if self.alpha > 0:
            probs = probs ** self.alpha
            probs /= probs.sum()
        else:
            probs = np.ones(n, dtype=np.float64) / n

        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self._frame / self.beta_frames)
        self._frame += 1

        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        transitions = [self._buffer[i] for i in indices]
        return transitions, indices.tolist(), weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: List[float]) -> None:
        for i, err in zip(indices, td_errors):
            if i < len(self._priorities):
                priority = abs(err) + 1e-5
                self._priorities[i] = priority
                self._max_priority = max(self._max_priority, priority)

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        return len(self._buffer) >= 32
