"""
Double DQN agent for sprint planning.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from env.models import Action, Observation
from .heuristic_agent import HeuristicAgent
from .features import action_embedding, action_space, encode, valid_action_embeddings
from .q_network import QNetwork
from .replay_buffer import ReplayBuffer, Transition

logger = logging.getLogger(__name__)


class DDQNAgent:
    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        decay_steps: int = 5_000,
        batch_size: int = 64,
        target_update: int = 200,
        buffer_size: int = 10_000,
        guided_exploration_prob: float = 0.65,
        seed: int = 42,
    ):
        random.seed(seed)
        np.random.seed(seed)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = decay_steps
        self.batch_size = batch_size
        self.target_update = target_update
        self.seed = seed
        self.guided_exploration_prob = guided_exploration_prob

        self.online = QNetwork(lr=lr, seed=seed)
        self.target = QNetwork(lr=lr, seed=seed)
        self.target.copy_weights_from(self.online)

        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.heuristic = HeuristicAgent()

        self.total_steps = 0
        self.train_steps = 0
        self.episode = 0
        self.loss_history: List[float] = []

    def act(self, obs: Observation, deterministic: bool = False) -> Optional[Action]:
        actions = action_space(obs)
        if not actions:
            return None

        if not deterministic and random.random() < self.epsilon:
            task_id, dev_id = self._explore_action(obs, actions)
        else:
            task_id, dev_id = self._greedy_action(obs, actions)
        return Action(task_id=task_id, developer_id=dev_id)

    def _explore_action(self, obs: Observation, actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        if random.random() < self.guided_exploration_prob:
            suggested = self.heuristic.act(obs)
            if suggested is not None:
                chosen = (suggested.task_id, suggested.developer_id)
                if chosen in actions:
                    return chosen
        return random.choice(actions)

    def observe(
        self,
        state: np.ndarray,
        action_features: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_valid_action_features: List[np.ndarray],
        done: bool,
        task_id: str = "",
        dev_id: str = "",
    ) -> None:
        self.buffer.push(
            Transition(
                state=state,
                action_features=action_features,
                reward=reward,
                next_state=next_state,
                next_valid_action_features=next_valid_action_features,
                done=done,
                task_id=task_id,
                dev_id=dev_id,
            )
        )

        self.total_steps += 1
        self._decay_epsilon()

        if self.buffer.is_ready:
            loss = self._learn()
            self.loss_history.append(loss)

        if self.total_steps % self.target_update == 0:
            self.target.copy_weights_from(self.online)

    def _greedy_action(self, obs: Observation, actions: List[Tuple[str, str]]) -> Tuple[str, str]:
        state_vec = encode(obs)
        best_q = -np.inf
        best_action = actions[0]

        for task_id, dev_id in actions:
            emb = action_embedding(obs, task_id, dev_id)
            x = np.concatenate([state_vec, emb])
            q = self.online.predict(x)
            if q > best_q:
                best_q = q
                best_action = (task_id, dev_id)
        return best_action

    def _learn(self) -> float:
        transitions, indices, weights = self.buffer.sample(self.batch_size)
        batch_x = []
        batch_target = []
        td_errors = []

        for tr in transitions:
            x = np.concatenate([tr.state, tr.action_features])
            q_now = self.online.predict(x)

            if tr.done or not tr.next_valid_action_features:
                target = tr.reward
            else:
                best_next_emb = self._best_valid_next_emb(tr.next_state, tr.next_valid_action_features)
                q_next = self.target.predict(np.concatenate([tr.next_state, best_next_emb]))
                target = tr.reward + self.gamma * q_next

            td_errors.append(target - q_now)
            batch_x.append(x)
            batch_target.append([target])

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_target = np.array(batch_target, dtype=np.float32)
        loss = self.online.update(batch_x, batch_target, sample_weights=weights)
        self.buffer.update_priorities(indices, [abs(err) for err in td_errors])
        self.train_steps += 1
        return loss

    def _best_valid_next_emb(self, next_state: np.ndarray, valid_action_embs: List[np.ndarray]) -> np.ndarray:
        best_q = -np.inf
        best_emb = valid_action_embs[0]
        for emb in valid_action_embs:
            q = self.online.predict(np.concatenate([next_state, emb]))
            if q > best_q:
                best_q = q
                best_emb = emb
        return best_emb

    def _decay_epsilon(self) -> None:
        frac = min(self.total_steps / self.decay_steps, 1.0)
        self.epsilon = self.epsilon_start - frac * (self.epsilon_start - self.epsilon_end)

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.online.save(f"{directory}/online.pkl")
        self.target.save(f"{directory}/target.pkl")
        meta = {
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "episode": self.episode,
            "epsilon": self.epsilon,
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
            self.episode = meta.get("episode", 0)
            self.epsilon = meta.get("epsilon", self.epsilon_end)
        logger.info(f"Agent loaded from {directory}")

    @property
    def stats(self) -> dict:
        recent_loss = self.loss_history[-100:] if self.loss_history else [0.0]
        return {
            "total_steps": self.total_steps,
            "train_steps": self.train_steps,
            "episode": self.episode,
            "epsilon": round(self.epsilon, 4),
            "buffer_size": len(self.buffer),
            "mean_loss_100": round(float(np.mean(recent_loss)), 5),
        }
