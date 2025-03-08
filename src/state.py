"""Module defining the State of the Graph."""

# Import packages and modules

from typing import TypedDict


# Define state
class GraphState(TypedDict):
    """
    State of the Graph.

    :param question: question asked by the user
    :type question: str
    :param generation: generation of the LLM
    :type generation: str
    :param documents: documents retrieved
    :type documents: list[str]
    :param confidence_score: confidence score of the retrieved documents
    :type confidence_score: float
    :pram iterations: number of generations iterations
    :type iterations: int
    """

    question: str
    generation: str
    documents: list[str]
    confidence_score: float
    iterations: int
