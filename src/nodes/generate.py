"""Module containing the Generation node."""

# Import packages and modules
import warnings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_ollama import ChatOllama

from src.state import GraphState
from src.chains.generation import generation_chain
from src.utils.logging import logger

warnings.filterwarnings("ignore")

# Define contextual compressor
llm = ChatOllama(
    model="mistral",
    temperature=0.0,
)
compressor = LLMChainExtractor.from_llm(llm)


# Define the Generation node
def generation_node(
    state: GraphState,
) -> dict[str, any]:
    """
    Function defining the Generation node.

    It invokes the generation chain with the context and question from the state.
    :param state: state of the Graph
    :type state: GraphState
    :return: answer to the question with updated state
    :rtype: dict[str, any]
    """
    logger.info("Compressing documents...")
    compressed_docs = compressor.compress_documents(
        documents=state.get("documents"),
        query=state.get("question"),
    )
    logger.info(f"Compressed {len(state.get("documents"))} documents into {len(compressed_docs)} documents.")
    logger.info("Generating answer...")
    generation = generation_chain.invoke(
        {
            "context": compressed_docs, #state.get("documents"),
            "question": state.get("question"),
        }
    )

    return {
        "question": state.get("question"),
        "documents": compressed_docs, #state.get("documents"),
        "generation": generation,
        "iterations": (state.get("iterations", 0) + 1),
    }
