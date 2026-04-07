"""
Double Deep Q-Network (DDQN) Agent for Sprint Planning.

Key design choices:
  ─ Double DQN:        online net selects action, target net evaluates it
                       → removes overestimation bias
  ─ Experience Replay: uniform buffer (PER available via alpha > 0)
  ─ ε-greedy decay:    linear from ε_start → ε_end over `decay_steps`
  ─ Target network:    hard-copied every `target_update` steps
  ─ Batch norm:        not used (overkill for this problem size)

Action representation:
  All valid (task_id, dev_id) pairs are enumerated each step.
  We score each pair by running a forward pass of Q(s, action_embedding).
  The agent picks the argmax (greedy) or samples uniformly (exploration).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from env.models import Observation, Action
from .features import encode, action_space, FEATURE_DIM, MAX_TASKS, MAX_DEVS
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)

ACTION_DIM = 2   # (task_idx_norm, dev_idx_norm)


class DDQNAgent:
    """
    Double DQN agent for the SprintEnv task-assignment problem.

    Args:
        lr:             Learning rate for Adam.
        gamma:          Discount factor.
        epsilon_start:  Initial exploration rate.
        epsilon_end:    Final exploration rate.
        decay_steps:    Steps over which epsilon decays linearly.
        batch_size:     Replay batch size.
        target_update:  Hard-update target net every N steps.
        buffer_size:    Replay buffer capacity.
        seed:           Random seed.
    """

    def __init__(
        self,
        lr:             float = 1e-3,
        gamma:          float = 0.95,
        epsilon_start:  float = 1.0,
        epsilon_end:    float = 0.05,
        decay_steps:    int   = 5_000,
        batch_size:     int   = 64,
        target_update:  int   = 200,
        buffer_size:    int   = 10_000,
        seed:           int   = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.gamma          = gamma
        self.epsilon        = epsilon_start
        self.epsilon_start  = epsilon_start
        self.epsilon_end    = epsilon_end
        self.decay_steps    = decay_steps
        self.batch_size     = batch_size
        self.target_update  = target_update
        self.seed           = seed

        # Networks
        self.online  = QNetwork(lr=lr, seed=seed)
        self.target  = QNetwork(lr=lr, seed=seed)
        self.target.copy_weights_from(self.online)

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_size)

        # Counters
        self.total_steps  = 0
        self.train_steps  = 0
        self.episode      = 0

        # Loss history
        self.loss_history: List[float] = []

    # ── Core agent interface ─────────────────────────────────────────────────

    def act(self, obs: Observation, deterministic: bool = False) -> Optional[Action]:
        """
        Choose an action given the current observation.

        Returns None if no valid actions exist (all devs over-capacity).
        Uses ε-greedy when not deterministic.
        """
        actions = action_space(obs)
        if not actions:
            return None

        if not deterministic and random.random() < self.epsilon:
            # Explore: random valid action
            task_id, dev_id = random.choice(actions)
        else:
            # Exploit: argmax Q(s, a)
            task_id, dev_id = self._greedy_action(obs, actions)

        return Action(task_id=task_id, developer_id=dev_id)

    def observe(
        self,
        state:      np.ndarray,
        action_idx: int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
        task_id:    str = "",
        dev_id:     str = "",
    ) -> None:
        """Store transition and trigger learning if buffer is ready."""
        self.buffer.push(Transition(
            state=state,
            action_idx=action_idx,
            reward=reward,
            next_state=next_state,
            done=done,
            task_id=task_id,
            dev_id=dev_id,
        ))

        self.total_steps += 1
        self._decay_epsilon()

        if self.buffer.is_ready:
            loss = self._learn()
            self.loss_history.append(loss)

        if self.total_steps % self.target_update == 0:
            self.target.copy_weights_from(self.online)
            logger.debug(f"Target network synced at step {self.total_steps}")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _greedy_action(
        self,
        obs: Observation,
        actions: List[Tuple[str, str]],
    ) -> Tuple[str, str]:
        """Return (task_id, dev_id) with highest estimated Q-value."""
        state_vec = encode(obs)
        all_tasks = obs.jira_tickets
        all_devs  = obs.developers

        best_q     = -np.inf
        best_action = actions[0]

        for (task_id, dev_id) in actions:
            # Build (task_idx_norm, dev_idx_norm) embedding
            t_idx = next((i for i, t in enumerate(all_tasks) if t.id == task_id), 0)
            d_idx = next((i for i, d in enumerate(all_devs)  if d.id == dev_id),  0)
            action_emb = np.array(
                [t_idx / max(len(all_tasks) - 1, 1),
                 d_idx / max(len(all_devs)  - 1, 1)],
                dtype=np.float32
            )
            x = np.concatenate([state_vec, action_emb])
            q = self.online.predict(x)

            if q > best_q:
                best_q      = q
                best_action = (task_id, dev_id)

        return best_action

    def _learn(self) -> float:
        """Sample a batch and do one gradient update (Double DQN)."""
        transitions, indices, weights = self.buffer.sample(self.batch_size)

        # Build batch arrays
        batch_x      = []
        batch_target = []
        td_errors    = []

        for tr in transitions:
            s  = tr.state
            ns = tr.next_state
            r  = tr.reward
            d  = tr.done

            # Current Q: we stored action_idx but don't strictly need it for the update
            # We reconstruct the input with action embedding via action_idx
            a_emb = self._idx_to_emb(tr.action_idx)
            x     = np.concatenate([s, a_emb])
            q_now = self.online.predict(x)

            # Double DQN target ──────────────────────────────────────────────
            if d:
                target = r
            else:
                # 1. Online net picks best next action
                best_next_emb = self._best_next_emb(ns)
                x_next_online  = np.concatenate([ns, best_next_emb])
                # 2. Target net evaluates it
                x_next_target  = np.concatenate([ns, best_next_emb])
                q_next = self.target.predict(x_next_target)
                target = r + self.gamma * q_next

            td_errors.append(target - q_now)
            batch_x.append(x)
            batch_target.append([target])

        batch_x      = np.array(batch_x,      dtype=np.float32)
        batch_target = np.array(batch_target, dtype=np.float32)
        mean_weight  = float(np.mean(weights))

        loss = self.online.update(batch_x, batch_target, weight=mean_weight)
        self.buffer.update_priorities(indices, [abs(e) for e in td_errors])
        self.train_steps += 1

        return loss

    def _idx_to_emb(self, action_idx: int) -> np.ndarray:
        """Decode stored action index back to normalised 2D embedding."""
        t_idx = action_idx // MAX_DEVS
        d_idx = action_idx %  MAX_DEVS
        return np.array(
            [t_idx / max(MAX_TASKS - 1, 1),
             d_idx / max(MAX_DEVS  - 1, 1)],
            dtype=np.float32
        )

    def _best_next_emb(self, next_state: np.ndarray) -> np.ndarray:
        """
        Find the action embedding that maximises Q(next_state, ·) using the online network.
        Scans all (task_idx, dev_idx) combinations.
        """
        best_q   = -np.inf
        best_emb = np.zeros(ACTION_DIM, dtype=np.float32)

        for t_idx in range(MAX_TASKS):
            for d_idx in range(MAX_DEVS):
                emb = np.array(
                    [t_idx / max(MAX_TASKS - 1, 1),
                     d_idx / max(MAX_DEVS  - 1, 1)],
                    dtype=np.float32
                )
                x = np.concatenate([next_state, emb])
                q = self.online.predict(x)
                if q > best_q:
                    best_q   = q
                    best_emb = emb

        return best_emb

    def _decay_epsilon(self) -> None:
        frac = min(self.total_steps / self.decay_steps, 1.0)
        self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    # ── Encode action for storage ────────────────────────────────────────────

    def encode_action(self, obs: Observation, task_id: str, dev_id: str) -> int:
        """Encode (task_id, dev_id) as a flat integer for replay storage."""
        all_tasks = obs.jira_tickets
        all_devs  = obs.developers
        t_idx = next((i for i, t in enumerate(all_tasks) if t.id == task_id), 0)
        d_idx = next((i for i, d in enumerate(all_devs)  if d.id == dev_id),  0)
        # Clamp to MAX bounds
        t_idx = min(t_idx, MAX_TASKS - 1)
        d_idx = min(d_idx, MAX_DEVS  - 1)
        return t_idx * MAX_DEVS + d_idx

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.online.save(f"{directory}/online.pkl")
        self.target.save(f"{directory}/target.pkl")
        meta = {
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "episode":     self.episode,
            "epsilon":     self.epsilon,
        }
        (Path(directory) / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info(f"Agent saved to {directory}")

    def load(self, directory: str) -> None:
        self.online.load(f"{directory}/online.pkl")
        self.target.load(f"{directory}/target.pkl")
        meta_path = Path(directory) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.total_steps = meta.get("total_steps", 0)
            self.train_steps = meta.get("train_steps", 0)
            self.episode     = meta.get("episode", 0)
            self.epsilon     = meta.get("epsilon", self.epsilon_end)
        logger.info(f"Agent loaded from {directory}")

    @property
    def stats(self) -> dict:
        recent_loss = self.loss_history[-100:] if self.loss_history else [0]
        return {
            "total_steps":  self.total_steps,
            "train_steps":  self.train_steps,
            "episode":      self.episode,
            "epsilon":      round(self.epsilon, 4),
            "buffer_size":  len(self.buffer),
            "mean_loss_100": round(float(np.mean(recent_loss)), 5),
        }