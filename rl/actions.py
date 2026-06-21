from dataclasses import dataclass


@dataclass(frozen=True)
class SearchAction:
    key: str
    label: str
    crossover_rate: float
    mutation_rate: float
    mutation: str
    local_search: bool = False
    cost: float = 0.0


SEARCH_ACTIONS = (
    SearchAction("exploit", "High crossover / low mutation", 0.90, 0.08, "swap"),
    SearchAction("balanced", "Balanced GA", 0.80, 0.18, "swap"),
    SearchAction("explore", "Exploration mutation", 0.65, 0.28, "swap"),
    SearchAction(
        "insertion",
        "Insertion mutation",
        0.75,
        0.22,
        "insert",
        cost=0.02,
    ),
    SearchAction(
        "bottleneck_refine",
        "Bottleneck local search",
        0.80,
        0.14,
        "swap",
        local_search=True,
        cost=0.05,
    ),
)
