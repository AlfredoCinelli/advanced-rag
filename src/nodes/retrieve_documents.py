"""Module containing the Retriever node."""

# Import packages and modules

from src.state import GraphState

from langchain_core.documents import Document
import warnings

from src.retriever import retriever
from src.utils.logging import logger

warnings.filterwarnings("ignore")

# Define Retriever Node


def retriever_node(
    state: GraphState,
) -> dict[str, str | list[Document] | Document]:
    """
    Function defining the Retriever node.

    :param state: state of the Graph
    :type state: GraphState
    :return: dictionary containing the question and the documents retrieved
    :rtype: dict[str, str | list[Document] | Document]
    """
    logger.info("Performing retrieval...")

    documents = retriever.invoke(
        state.get("question"),
    )
    logger.info(f"Retrieved {len(documents)} documents.")
    return {
        "question": state.get("question"),
        "documents": documents,
    }
