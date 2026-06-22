from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Mapping

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
        self._state_shape: tuple[int, ...] | None = None

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
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)
        if state_array.shape != next_state_array.shape:
            raise ValueError(
                "State and next_state must have matching shapes, got "
                f"{state_array.shape} and {next_state_array.shape}"
            )
        if self._state_shape is None:
            self._state_shape = state_array.shape
        elif state_array.shape != self._state_shape:
            raise ValueError(
                f"Transition state shape must be the expected {self._state_shape}, "
                f"got {state_array.shape}"
            )
        self._transitions.append(
            (
                state_array.copy(),
                int(action),
                float(reward),
                next_state_array.copy(),
                bool(done),
            )
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": self._transitions.maxlen,
            "state_shape": self._state_shape,
            "transitions": [
                (
                    state.tolist(),
                    action,
                    reward,
                    next_state.tolist(),
                    done,
                )
                for state, action, reward, next_state, done in self._transitions
            ],
            "random_state": self._random.getstate(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        capacity = int(state["capacity"])
        if capacity <= 0:
            raise ValueError("Saved replay capacity must be positive")

        raw_shape = state["state_shape"]
        state_shape = (
            None if raw_shape is None else tuple(int(value) for value in raw_shape)
        )
        raw_transitions = list(state["transitions"])
        if len(raw_transitions) > capacity:
            raise ValueError(
                "Saved replay transition count exceeds its capacity: "
                f"{len(raw_transitions)} > {capacity}"
            )
        if raw_transitions and state_shape is None:
            raise ValueError("Saved replay with transitions must include state_shape")

        self._transitions = deque(maxlen=capacity)
        self._state_shape = state_shape
        for raw_transition in raw_transitions:
            if len(raw_transition) != 5:
                raise ValueError("Saved replay transitions must contain five values")
            raw_state, action, reward, raw_next_state, done = raw_transition
            state_array = np.asarray(raw_state, dtype=np.float32)
            next_state_array = np.asarray(raw_next_state, dtype=np.float32)
            if state_shape is not None:
                if (
                    state_array.shape != state_shape
                    or next_state_array.shape != state_shape
                ):
                    raise ValueError(
                        "Saved replay transition shape does not match state_shape "
                        f"{state_shape}"
                    )
            self.push(state_array, action, reward, next_state_array, done)
        self._random.setstate(state["random_state"])

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
