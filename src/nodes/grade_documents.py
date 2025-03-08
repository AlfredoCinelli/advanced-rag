"""Module containing the Document Grader node."""

# Import packages and modules
import warnings

from langchain_core.documents import Document

from src.state import GraphState
from src.chains.retrieval_grader import retrieval_grader
from src.utils.logging import logger

warnings.filterwarnings("ignore")


# Define the document grader Node
def grader_node(state: GraphState) -> dict[str, any]:
    """
    Function defining the documents grader node.
    It iterates over all the retrieved documents and grades them using the retrieval grader chain.
    If the document is relevant, it is added to the filtered documents list.
    If the document is not relevant, it is added to the filtered documents list and the web search flag is set to True.

    :param state: state of the Graph
    :type state: GraphState
    :return: dictionary containing the question and the documents retrieved
    :rtype: dict[str, any]
    """

    logger.info("Grading documents...")
    question: str = state.get("question")
    documents: list[Document] | Document = (
        state.get("documents")
        if isinstance(state.get("documents"), list)
        else [state.get("documents")]
    )

    filtered_docs: list[Document] = []  # documents that are relevant to the question

    for document in documents:
        score: str = retrieval_grader.invoke(
            {
                "question": question,
                "document": document.page_content,
            }
        )

        if score.binary_score.lower() == "yes":
            logger.info("Document is relevant.")
            filtered_docs.append(document)
        elif score.binary_score.lower() == "no":
            logger.info("Document is not relevant.")
            continue
    confidence_score = round(len(filtered_docs) / len(documents), 4)
    logger.info(
        f"Graded {len(filtered_docs)} documents as relevant with a confidence score of {confidence_score:.2%}"
    )

    return {
        "question": question,
        "documents": filtered_docs,
        "confidence_score": confidence_score,
    }
