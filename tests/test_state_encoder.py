from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from rl.state_encoder import StateContext, encode_state


POPULATION = [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]]
CONTEXT = StateContext(
    initial_best=100.0,
    initial_average=120.0,
    generation=5,
    generation_budget=20,
    stagnation=2,
    remaining_ratio=0.75,
    machine_loads=(10.0, 20.0),
    breakdown_pressure=0.2,
    emergency_job=True,
)


def test_state_encoder_returns_the_expected_finite_ten_feature_vector():
    state = encode_state([90.0, 100.0, 110.0], POPULATION, CONTEXT)

    expected = np.array(
        [
            0.1,
            1.0 / 6.0,
            0.1,
            1.0,
            np.std([90.0, 100.0, 110.0]) / 100.0,
            0.1,
            0.25,
            0.75,
            np.std([10.0, 20.0]) / 15.0,
            0.44,
        ],
        dtype=np.float32,
    )

    assert state.shape == (10,)
    assert state.dtype == np.float32
    assert np.isfinite(state).all()
    assert ((state >= -1.0) & (state <= 1.0)).all()
    np.testing.assert_allclose(state, expected, rtol=1e-6, atol=1e-7)


def test_state_encoder_is_deterministic():
    first = encode_state([90.0, 100.0, 110.0], POPULATION, CONTEXT)
    second = encode_state([90.0, 100.0, 110.0], POPULATION, CONTEXT)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("machine_loads", [(), (0.0, 0.0)])
def test_machine_load_variation_is_zero_without_positive_loads(machine_loads):
    context = replace(CONTEXT, machine_loads=machine_loads)

    state = encode_state([90.0, 100.0, 110.0], POPULATION, context)

    assert state[8] == 0.0


@pytest.mark.parametrize(
    ("fitness_values", "population"),
    [([], POPULATION), ([1.0], [])],
)
def test_state_encoder_rejects_empty_fitness_or_population(
    fitness_values, population
):
    with pytest.raises(ValueError):
        encode_state(fitness_values, population, CONTEXT)


@pytest.mark.parametrize(
    ("fitness_values", "population"),
    [
        ([1.0, np.nan], POPULATION),
        ([1.0, np.inf], POPULATION),
        ([1.0], [[0.0, np.nan]]),
        ([1.0], [[0.0, np.inf]]),
    ],
)
def test_state_encoder_rejects_non_finite_fitness_or_population(
    fitness_values, population
):
    with pytest.raises(ValueError):
        encode_state(fitness_values, population, CONTEXT)


@pytest.mark.parametrize(
    "context",
    [
        replace(CONTEXT, initial_best=np.nan),
        replace(CONTEXT, initial_average=np.inf),
        replace(CONTEXT, generation=np.nan),
        replace(CONTEXT, generation_budget=np.inf),
        replace(CONTEXT, stagnation=np.nan),
        replace(CONTEXT, remaining_ratio=np.inf),
        replace(CONTEXT, machine_loads=(10.0, np.nan)),
        replace(CONTEXT, breakdown_pressure=np.inf),
    ],
)
def test_state_encoder_rejects_non_finite_context(context):
    with pytest.raises(ValueError):
        encode_state([90.0, 100.0, 110.0], POPULATION, context)


def test_state_context_is_frozen():
    with pytest.raises(FrozenInstanceError):
        CONTEXT.remaining_ratio = 0.5
