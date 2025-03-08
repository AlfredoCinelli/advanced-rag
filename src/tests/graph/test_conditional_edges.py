# Import packages and modules

import pytest
from unittest.mock import patch, MagicMock
from src.state import GraphState

from src.graph import (
    route_question,
    decide_to_generate,
    grade_generation,
)
from src.tests.graph.data import scenario

# Define test functions
@pytest.mark.parametrize("scenario", scenario("route_question"))
def test_route_question(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test route question chain."""
    with patch(
        target="src.graph.question_router",
        new=mocked_chain(chain=scenario.get("chain_name")),
    ):
        res = route_question(state=state(scenario.get("state_name")))
        assert res == scenario.get("expected_output")

@pytest.mark.parametrize("scenario", scenario("decide_to_generate"))
def test_decide_to_generate(
    state: GraphState,
    scenario: dict[str, any],
) -> None:
    """Test decide to generate chain."""
    res = decide_to_generate(state=state(scenario.get("state_name")))
    assert res == scenario.get("expected_output")

@pytest.mark.parametrize("scenario", scenario("grade_generation"))
def test_grade_generation(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test grade generation chain."""
    if scenario.get("max_iterations", False) is True:
        res = grade_generation(state=state(scenario.get("state_name")))
        assert res == scenario.get("expected_output")
    else:
        with (
            patch(
                target="src.graph.hallucination_grader",
                new=mocked_chain(chain=scenario.get("hallucination_chain_name")) if scenario.get("hallucination_chain_name") else None,
            ),
            patch(
                target="src.graph.answer_grader",
                new=mocked_chain(chain=scenario.get("answer_chain_name")) if scenario.get("answer_chain_name") else None,
            ),
        ):
            res = grade_generation(state=state(scenario.get("state_name")))
            assert res == scenario.get("expected_output")