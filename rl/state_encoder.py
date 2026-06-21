from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateContext:
    initial_best: float
    initial_average: float
    generation: int
    generation_budget: int
    stagnation: int
    remaining_ratio: float
    machine_loads: tuple[float, ...]
    breakdown_pressure: float
    emergency_job: bool


def _finite_array(values, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    return array


def _stable_mean(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return float(np.mean(values / scale) * scale)


def _stable_std(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    return float(np.std(values / scale) * scale)


def _guarded_ratio(numerator: float, denominator: float) -> float:
    guarded_denominator = max(abs(denominator), np.finfo(np.float64).eps)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        ratio = np.divide(numerator, guarded_denominator)
    return float(np.clip(ratio, -1.0, 1.0))


def _population_diversity(population) -> float:
    chromosomes = list(population)
    if not chromosomes:
        raise ValueError("population must not be empty")

    unique_chromosomes = set()
    for chromosome in chromosomes:
        values = _finite_array(chromosome, "population")
        key = (values.shape, tuple(values.reshape(-1).tolist()))
        unique_chromosomes.add(key)
    return len(unique_chromosomes) / len(chromosomes)


def encode_state(fitness_values, population, context: StateContext) -> np.ndarray:
    fitness = _finite_array(fitness_values, "fitness_values").reshape(-1)
    if fitness.size == 0:
        raise ValueError("fitness_values must not be empty")

    diversity = _population_diversity(population)
    _finite_array(
        (
            context.initial_best,
            context.initial_average,
            context.generation,
            context.generation_budget,
            context.stagnation,
            context.remaining_ratio,
            context.breakdown_pressure,
            context.emergency_job,
        ),
        "context",
    )
    machine_loads = _finite_array(context.machine_loads, "machine_loads").reshape(-1)

    current_best = float(np.min(fitness))
    current_average = _stable_mean(fitness)
    fitness_std = _stable_std(fitness)

    if machine_loads.size == 0 or np.all(machine_loads == 0.0):
        load_variation = 0.0
    else:
        load_variation = _guarded_ratio(
            _stable_std(machine_loads), _stable_mean(machine_loads)
        )

    disturbance = float(
        np.clip(
            0.7 * context.breakdown_pressure + 0.3 * int(context.emergency_job),
            0.0,
            1.0,
        )
    )
    features = np.array(
        [
            _guarded_ratio(
                context.initial_best - current_best, context.initial_best
            ),
            _guarded_ratio(
                context.initial_average - current_average,
                context.initial_average,
            ),
            _guarded_ratio(current_average - current_best, current_average),
            diversity,
            _guarded_ratio(fitness_std, current_average),
            _guarded_ratio(context.stagnation, context.generation_budget),
            _guarded_ratio(context.generation, context.generation_budget),
            context.remaining_ratio,
            load_variation,
            disturbance,
        ],
        dtype=np.float64,
    )
    state = np.clip(features, -1.0, 1.0).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError("encoded state must be finite")
    return state
