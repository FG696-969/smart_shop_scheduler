import numpy as np
import pytest

from rl.replay_buffer import ReplayBuffer


def test_replay_buffer_is_bounded_and_copies_pushed_arrays():
    buffer = ReplayBuffer(capacity=2, seed=7)
    state = np.zeros(10, dtype=np.float32)
    next_state = np.ones(10, dtype=np.float32)

    buffer.push(state, 1, 0.5, next_state, False)
    state[:] = 9.0
    next_state[:] = 8.0
    copied = buffer.sample(1)

    np.testing.assert_array_equal(copied.states[0], np.zeros(10))
    np.testing.assert_array_equal(copied.next_states[0], np.ones(10))

    buffer.push(np.full(10, 2.0, dtype=np.float32), 2, 1.0, next_state, True)
    buffer.push(np.full(10, 3.0, dtype=np.float32), 3, -1.0, next_state, False)

    assert len(buffer) == 2


def test_replay_buffer_sample_has_expected_shapes_and_dtypes():
    buffer = ReplayBuffer(capacity=4, seed=3)
    for index in range(4):
        state = np.full(10, index, dtype=np.float32)
        buffer.push(state, index % 5, float(index), state + 1, index == 3)

    batch = buffer.sample(3)

    assert batch.states.shape == (3, 10)
    assert batch.next_states.shape == (3, 10)
    assert batch.actions.shape == (3,)
    assert batch.rewards.shape == (3,)
    assert batch.dones.shape == (3,)
    assert batch.states.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == np.float32


def test_replay_buffer_rejects_insufficient_sample():
    buffer = ReplayBuffer(capacity=4, seed=3)
    buffer.push(np.zeros(10, dtype=np.float32), 0, 0.0, np.ones(10), False)

    with pytest.raises(ValueError, match="Cannot sample 2 transitions"):
        buffer.sample(2)


def test_replay_buffer_rejects_inconsistent_transition_shapes():
    buffer = ReplayBuffer(capacity=4, seed=3)

    with pytest.raises(ValueError, match="matching shapes"):
        buffer.push(np.zeros(10), 0, 0.0, np.zeros(9), False)

    buffer.push(np.zeros(10), 0, 0.0, np.ones(10), False)
    with pytest.raises(ValueError, match=r"expected \(10,\)"):
        buffer.push(np.zeros(8), 0, 0.0, np.ones(8), False)


def test_replay_buffer_state_round_trip_preserves_content_metadata_and_rng():
    source = ReplayBuffer(capacity=5, seed=17)
    for index in range(4):
        state = np.full(3, index, dtype=np.float32)
        source.push(state, index, index / 10, state + 1, index == 3)

    saved_state = source.state_dict()
    restored = ReplayBuffer(capacity=1, seed=99)
    restored.load_state_dict(saved_state)

    restored_state = restored.state_dict()
    assert restored_state["capacity"] == 5
    assert restored_state["state_shape"] == (3,)
    assert restored_state["transitions"] == saved_state["transitions"]
    expected = source.sample(3)
    actual = restored.sample(3)
    np.testing.assert_array_equal(actual.states, expected.states)
    np.testing.assert_array_equal(actual.actions, expected.actions)
    np.testing.assert_array_equal(actual.rewards, expected.rewards)
    np.testing.assert_array_equal(actual.next_states, expected.next_states)
    np.testing.assert_array_equal(actual.dones, expected.dones)
