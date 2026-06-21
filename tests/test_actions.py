from dataclasses import FrozenInstanceError

import pytest

from rl.actions import SEARCH_ACTIONS, SearchAction


def test_action_registry_has_the_exact_shared_search_actions():
    assert isinstance(SEARCH_ACTIONS, tuple)
    assert SEARCH_ACTIONS == (
        SearchAction(
            "exploit",
            "High crossover / low mutation",
            0.90,
            0.08,
            "swap",
        ),
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
    assert all(isinstance(action, SearchAction) for action in SEARCH_ACTIONS)
    assert [action.mutation for action in SEARCH_ACTIONS] == [
        "swap",
        "swap",
        "swap",
        "insert",
        "swap",
    ]
    assert [action.local_search for action in SEARCH_ACTIONS] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_search_action_is_frozen():
    with pytest.raises(FrozenInstanceError):
        SEARCH_ACTIONS[0].mutation_rate = 0.5
