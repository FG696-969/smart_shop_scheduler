from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np


@dataclass(frozen=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive")
        self._transitions: Deque[
            tuple[np.ndarray, int, float, np.ndarray, bool]
        ] = deque(maxlen=capacity)
        self._random = random.Random(seed)

    def __len__(self) -> int:
        return len(self._transitions)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self._transitions.append(
            (
                np.asarray(state, dtype=np.float32).copy(),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32).copy(),
                bool(done),
            )
        )

    def sample(self, batch_size: int) -> TransitionBatch:
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if len(self) < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {len(self)}"
            )
        transitions = self._random.sample(list(self._transitions), batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        return TransitionBatch(
            states=np.stack(states).astype(np.float32, copy=False),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states).astype(np.float32, copy=False),
            dones=np.asarray(dones, dtype=np.float32),
        )
