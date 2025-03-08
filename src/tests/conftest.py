"""Pytest configuration file with fixtures."""

# Import packages and modules

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.chains.retrieval_grader import GradeDocuments
from src.chains.router import RouteQuery
from src.chains.hallucination_grader import GradeHallucination
from src.chains.answer_grader import GradeAnswer
from src.state import GraphState

# Define fixtures
@pytest.fixture(scope="module")
def state() -> callable:
    """Fixture to mock the graph state."""
    def _mock_state(
        node: str,
    ) -> dict[str, str]:
        """
        Internal function to parametrize the fixture.

        :param node: name of the node to be tested
        :type node: str
        :return: mocked graph state
        :rtype: dict[str, str]
        """
        match node:
            case "generation_node":
                state: GraphState = {
                    "documents": [Document(page_content="This is context retrieved from the vector store or via web search.")],
                    "question": "User input question.",
                }
            case "grader_node":
                state: GraphState = {
                    "question": "User input question.",
                    "documents": [Document(page_content="This is the content of a document.", metadata={})],
                }
            case "retriever_node":
                state: GraphState = {
                    "question": "User input question.",
                }
            case "agent_search_node_no_documents":
                state: GraphState = {
                    "question": "User input question.",
                    "documents": None,
                }
            case "agent_search_node_documents":
                state: GraphState = {
                    "question": "User input question.",
                    "documents": [Document(page_content="This is the content of a document.", metadata={})],
                }
            case "route_question":
                state: GraphState = {
                    "question": "User input question.",
                }
            case "decide_to_generate_confidence":
                state: GraphState = {
                    "confidence_score": 0.8,
                }
            case "decide_to_generate_no_confidence":
                state: GraphState = {
                    "confidence_score": 0.2,
                }
            case "grade_generation_max_iters":
                state: GraphState = {
                    "iterations": 4,
                    "documents": [Document(page_content="This is the content of a document.", metadata={})],
                    "question": "User input question.",
                    "generation": "This is an LLM generated answer.",
                }
            case "grade_generation_lower_iters":
                state: GraphState = {
                    "iterations": 1,
                    "documents": [Document(page_content="This is the content of a document.", metadata={})],
                    "question": "User input question.",
                    "generation": "This is an LLM generated answer.",
                }
        return state
    return _mock_state


@pytest.fixture(scope="module")
def mocked_chain() -> callable:
    """Fixture to mock the LangChain chaings."""
    def _mock_chain(
        chain: str,
    ) -> MagicMock:
        """
        Internal function to parametrize the fixture.

        :param chain: name of the chain to be mocked
        :type chain: str
        :return: mocked LangChain chain with 'invoke' method
        :rtype: MagicMock
        """
        mocked_chain = MagicMock()
        match chain:
            case "generation_node":
                mocked_chain.invoke.return_value = "This is an LLM generated answer."
            case "grader_node_yes":
                mocked_chain.invoke.return_value = GradeDocuments(binary_score="yes")
            case "grader_node_no":
                mocked_chain.invoke.return_value = GradeDocuments(binary_score="no")
            case "retriever_node":
                mocked_chain.invoke.return_value = [Document(page_content="This is the content of a document.", metadata={})]
            case "agent_search_node":
                mocked_chain.invoke.return_value = {
                    "output": "This is the output of the agent search.",
                }
            case "route_question_vectorstore":
                mocked_chain.invoke.return_value = RouteQuery(data_source="vectorstore")
            case "route_question_websearch":
                mocked_chain.invoke.return_value = RouteQuery(data_source="websearch")
            case "hallucination_grader_hallucinate":
                mocked_chain.invoke.return_value = GradeHallucination(binary_score="no")
            case "hallucination_grader_no_hallucinate":
                mocked_chain.invoke.return_value = GradeHallucination(binary_score="yes")
            case "answer_factual":
                mocked_chain.invoke.return_value = GradeAnswer(binary_score="yes")
            case "answer_not_factual":
                mocked_chain.invoke.return_value = GradeAnswer(binary_score="no")
        
        return mocked_chain
    return _mock_chain