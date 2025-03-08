# Import packages and modules

import pytest
from unittest.mock import patch, MagicMock
from dotenv import load_dotenv
from langchain_core.documents import Document

from src.chains.retrieval_grader import (
    retrieval_grader,
    GradeDocuments,
)
from src.chains.generation import generation_chain
from src.retriever import retriever
from src.chains.tool_agent import (
    parse_tools,
)
from src.chains.hallucination_grader import hallucination_grader
from src.chains.answer_grader import answer_grader
from src.chains.router import question_router
from src.tests.chains.data import scenario

load_dotenv("local/.env")

# Test functions
@pytest.mark.parametrize("scenario", scenario("retrieval_grader"))
def test_retrieval_grader(scenario: dict[str, str]) -> None:
    """Test the retrieval grader chain."""
    res: GradeDocuments = retrieval_grader.invoke(
        {
            "question": scenario.get("question"),
            "document": scenario.get("document").page_content,
        }
    )
    print(res)
    assert res.binary_score == scenario.get("expected_output")


def test_retriever() -> None:
    """Test retriever component."""
    question = "agent memory"
    docs: list[Document] = retriever.invoke(question)
    assert (len(docs) == 5) and (isinstance(docs[0], Document))


@pytest.mark.parametrize("scenario", scenario("generation_chain"))
def test_generation_chain(scenario: dict[str, any]) -> None:
    """Test generation chain"""
    generation = generation_chain.invoke(
        {
            "context": scenario.get("documents"),
            "question": scenario.get("question"),
        }
    )
    assert isinstance(generation, str)


@pytest.mark.parametrize("scenario", scenario("parse_tools"))
def test_parse_tools(scenario: dict[str, any]) -> None:
    """Test parse tools."""
    tools = parse_tools(scenario.get("tools"))
    print(tools)
    assert tools == scenario.get("expected_output")


@pytest.mark.parametrize("scenario", scenario("tool_agent"))
def test_tool_agent(scenario: dict[str, any]) -> None:
    """Test tool agent executor."""
    with (
        patch(
            "src.chains.tool_agent.tool_agent_executor",
            new=MagicMock(
                invoke=MagicMock(return_value={"output": "This is an output"}),
            ),
        ),
    ):
        from src.chains.tool_agent import tool_agent_executor

        output = tool_agent_executor.invoke(input={"query": scenario.get("question")})
        assert output.get("output") == "This is an output"


@pytest.mark.parametrize("scenario", scenario("hallucination_grader"))
def test_hallucination_grader(
    scenario: dict[str, any],
) -> None:
    """Test hallucination grader."""
    res = hallucination_grader.invoke(
        {
            "documents": scenario.get("documents"),
            "generation": scenario.get("generation"),
        }
    )
    print(res)
    assert res.binary_score.lower() == scenario.get("expected_output")


@pytest.mark.parametrize("scenario", scenario("answer_grader"))
def test_answer_grader(
    scenario: dict[str, any],
) -> None:
    """Test answer grader."""
    res = answer_grader.invoke(
        {
            "question": scenario.get("question"),
            "generation": scenario.get("generation"),
        }
    )
    assert res.binary_score.lower() == scenario.get("expected_output")


@pytest.mark.parametrize("scenario", scenario("question_router"))
def test_question_router(
    scenario: dict[str, any],
) -> None:
    """Test question router."""
    res = question_router.invoke(
        {
            "question": scenario.get("question"),
        }
    )
    print(res)
    assert res.data_source == scenario.get("expected_output")
