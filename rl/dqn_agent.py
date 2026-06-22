from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

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
        torch.manual_seed(config.seed)
        self._random = random.Random(config.seed)
        self._np_random = np.random.RandomState(config.seed)
        self.device = torch.device(config.device)

        self.online = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target = QNetwork(config.state_size, config.action_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config.replay_capacity, config.seed)
        self.epsilon = float(config.epsilon_start)
        self.update_count = 0

    def state_dict(self) -> dict[str, object]:
        bit_generator, keys, position, has_gauss, cached_gaussian = (
            self._np_random.get_state()
        )
        state: dict[str, object] = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay_buffer": self.replay_buffer.state_dict(),
            "epsilon": self.epsilon,
            "update_count": self.update_count,
            "exploration_random_state": self._random.getstate(),
            "numpy_random_state": {
                "bit_generator": bit_generator,
                "keys": keys.tolist(),
                "position": position,
                "has_gauss": has_gauss,
                "cached_gaussian": cached_gaussian,
            },
            "torch_random_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda_random_state"] = torch.cuda.get_rng_state_all()
        return state

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_state[key] = value.to(self.device)
        self.replay_buffer.load_state_dict(state["replay_buffer"])
        self.epsilon = float(state["epsilon"])
        self.update_count = int(state["update_count"])
        self._random.setstate(state["exploration_random_state"])

        numpy_state = state["numpy_random_state"]
        self._np_random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(state["torch_random_state"].cpu())
        cuda_state = state.get("torch_cuda_random_state")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_state)

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
