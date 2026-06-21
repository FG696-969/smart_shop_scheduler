from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class DQNConfig:
    state_size: int = 10
    action_size: int = 5
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 10_000
    target_sync_interval: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42
    device: str = "cpu"


class QNetwork(nn.Module):
    def __init__(self, state_size: int = 10, action_size: int = 5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


class DQNAgent:
    def __init__(self, config: DQNConfig):
        if config.state_size <= 0 or config.action_size <= 0:
            raise ValueError("State and action sizes must be positive")
        if config.batch_size <= 0 or config.target_sync_interval <= 0:
            raise ValueError("Batch size and target sync interval must be positive")

        self.config = config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self._random = random.Random(config.seed)
        self.device = torch.device(config.device)

        self.online = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, config.seed)
        self.epsilon = float(config.epsilon_start)
        self.update_count = 0

    def _state_array(self, state: np.ndarray) -> np.ndarray:
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.config.state_size,):
            raise ValueError(
                f"State must have shape ({self.config.state_size},), got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError("State must contain finite values")
        return array

    def q_values(self, state: np.ndarray) -> np.ndarray:
        array = self._state_array(state)
        state_tensor = torch.from_numpy(array).to(self.device).unsqueeze(0)
        with torch.no_grad():
            values = self.online(state_tensor).squeeze(0)
        return values.detach().cpu().numpy().astype(np.float32, copy=True)

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        self._state_array(state)
        if explore and self._random.random() < self.epsilon:
            return self._random.randrange(self.config.action_size)
        return int(np.argmax(self.q_values(state)))

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        if not 0 <= int(action) < self.config.action_size:
            raise ValueError(f"Action must be in [0, {self.config.action_size})")
        self.replay_buffer.push(
            self._state_array(state),
            action,
            reward,
            self._state_array(next_state),
            done,
        )

    def train_step(self) -> float | None:
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = torch.from_numpy(batch.states).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device).unsqueeze(1)
        rewards = torch.from_numpy(batch.rewards).to(self.device)
        next_states = torch.from_numpy(batch.next_states).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)

        current_q = self.online(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(current_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=5.0)
        self.optimizer.step()

        self.update_count += 1
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )
        if self.update_count % self.config.target_sync_interval == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.detach().cpu().item())
