# Import packages and modules

import pytest
from unittest.mock import patch, MagicMock
from src.state import GraphState

from src.nodes import (
    generation_node,
    grader_node,
    retriever_node,
    agent_search_node,
)
from src.tests.nodes.data import scenario

# Define test functions
@pytest.mark.parametrize("scenario", scenario("generation_node"))
def test_generation_node(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test the generation node."""
    with patch(
        target="src.nodes.generate.generation_chain",
        new=mocked_chain(chain=scenario.get("chain_name")),
    ):
        res = generation_node(state(node=scenario.get("state_name")))
        assert res == scenario.get("expected_output")

@pytest.mark.parametrize("scenario", scenario("grader_node"))
def test_grader_node(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test the grader node."""
    with patch(
        target="src.nodes.grade_documents.retrieval_grader",
        new=mocked_chain(chain=scenario.get("chain_name")),
    ):
        res = grader_node(state(node=scenario.get("state_name")))
        assert res == scenario.get("expected_outout")

@pytest.mark.parametrize("scenario", scenario("retriever_node"))
def test_retriever_node(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test the retriever node."""
    with patch(
        target="src.nodes.retrieve_documents.retriever",
        new=mocked_chain(chain=scenario.get("chain_name")),
    ): 
        res = retriever_node(state(node=scenario.get("state_name")))
        assert res == scenario.get("expected_output")

@pytest.mark.parametrize("scenario", scenario("agent_search_node"))
def test_agent_search_node(
    state: GraphState,
    mocked_chain: MagicMock,
    scenario: dict[str, any],
) -> None:
    """Test the agent search node."""
    with patch(
        target="src.nodes.web_search.tool_agent_executor",
        new=mocked_chain(chain=scenario.get("chain_name")),
    ):
        res = agent_search_node(state(node=scenario.get("state_name")))
        print(res)
        assert res == scenario.get("expected_output")
