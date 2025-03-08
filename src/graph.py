"""
Module defining and orchestrating the Graph.

The Graph is an Advanced RAG that borrows structure and ideas from the followin RAGs:
- Corrective RAG (https://arxiv.org/abs/2401.15884).
- Self RAG (https://arxiv.org/abs/2310.11511).
- Adaptive RAG (https://arxiv.org/abs/2403.14403).

The vector store is implemented using Pinecone.
The embeddings are implemented using HuggingFace.
The chunking is based on Kamradt Semantic Chunking.
The main frameworks used are:
- LangChain.
- LangGraph.
- LangSmith.
"""

# Import packages and modules

import warnings
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import (
    StateGraph,
    END,
)
from src.constants import (
    GENERATE,
    GRADE_DOCUMENTS,
    RETRIEVE,
    WEBSEARCH,
    CONFIDENCE_THRESHOLD,
    MAX_ITERATIONS,
)
from src.nodes import (
    generation_node,
    grader_node,
    retriever_node,
    agent_search_node,
)
from src.chains.hallucination_grader import hallucination_grader
from src.chains.answer_grader import answer_grader
from src.chains.router import question_router
from src.state import GraphState
from src.utils.logging import logger

warnings.filterwarnings("ignore")
load_dotenv("local/.env")


# Define conditional edges
def decide_to_generate(
    state: GraphState,
) -> str:
    """
    Function defining the conditional edge and if to perform search or generate.

    :param state: state of the graph
    :type state: GraphState
    :return: action to be executed (mapped to the next node in the conditional edge)
    :rtype: str
    """
    logger.info("Deciding whether to generate or search...")
    confidence_score = state["confidence_score"]
    if confidence_score <= CONFIDENCE_THRESHOLD:
        logger.info(
            f"Retrieved documents are below the confidence score ({confidence_score} <= {CONFIDENCE_THRESHOLD})."
        )
        action = "not_correct"  # perform web search to extend knowledge/context
    else:
        logger.info(
            f"Retrieved documents are above the confidence score ({confidence_score} > {CONFIDENCE_THRESHOLD})."
        )
        action = "correct"  # generate answer (the retrieved context is good enough)
    return action


def grade_generation(
    state: GraphState,
) -> str:
    """
    Function defining the conditional edge performing critique/grading of the LLM generation.

    :param state: state of the graph
    :type state: GraphState
    :return: action to be executed (mapped to the next node in the conditional edge)
    :rtype: str
    """
    documents, question, generation, iterations = (
        state["documents"],
        state["question"],
        state["generation"],
        state["iterations"],
    )
    if iterations <= MAX_ITERATIONS:
        logger.info(f"Graph generated {iterations} times, still under the maximum iterations {MAX_ITERATIONS + 1}.")
        logger.info("Grading LLM answer compared to retrieved documents...")
        score = hallucination_grader.invoke(
            {
                "documents": documents,
                "generation": generation,
            }
        )
        if score.binary_score.lower() == "yes":
            logger.info("Answer is grounded in retrieved documents.")
            logger.info("Grading the LLM answer is factual to the question...")
            score = answer_grader.invoke(
                {
                    "question": question,
                    "generation": generation,
                }
            )
            if score.binary_score.lower() == "yes":
                # The answer is grounded in the documents and factual to the question, hence the answer is useful
                logger.info("Answer is factual to the question.")
                action = "useful"
            else:
                # The answer is grounded in the documents but not factual to the question, hence more knowledge is required
                logger.info("Answer is not factual to the question.")
                action = "not_useful"
        else:
            # The answer is not grounded in the documents, hence regenerating the answer is required (from the same retrieved context)
            logger.info("Answer is not grounded in retrieved documents.")
            action = "regen"
    else:
        # The Graph generation hits the maximum, exit with the latest generation/answer
        logger.warning(f"Graph generated {iterations} times, hitting the maximum iterations {MAX_ITERATIONS + 1}.")
        action = "stop"
    return action


def route_question(
    state: GraphState,
) -> str:
    """
    Function that defines a conditional edge that routes the question either to vector store or web search node.

    :param state: state of the graph
    :type state: GraphState
    :return: action to be executed (mapped to the next node in the conditional edge)
    :rtype: str
    """
    logger.info("Routing question to vector store or web search...")
    question = state["question"]
    res = question_router.invoke(
        {
            "question": question,
        }
    )
    if res.data_source == "websearch":
        logger.info("Routing question to web search...")
        action = "search"
    elif res.data_source == "vectorstore":
        logger.info("Routing question to vector store...")
        action = "vectorstore"
    return action


# Define Graph
workflow = StateGraph(GraphState)  # graph with custom state

# Define Nodes
workflow.add_node(RETRIEVE, retriever_node)
workflow.add_node(GRADE_DOCUMENTS, grader_node)
workflow.add_node(GENERATE, generation_node)
workflow.add_node(
    WEBSEARCH, agent_search_node
)  # web_search_node for deterministic Tavily search

# Define Edges
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
# Correction unit
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        "not_correct": WEBSEARCH,
        "correct": GENERATE,
    },
)
# Self-reflection unit
workflow.add_conditional_edges(
    GENERATE,
    grade_generation,
    path_map={
        "regen": GENERATE,
        "not_useful": WEBSEARCH,
        "useful": END,
        "stop": END,
    },
)
# Adaptive unit
workflow.set_conditional_entry_point(  # conditional entry point of the Graph (instead of deterministic entry point)
    route_question,
    path_map={
        "search": WEBSEARCH,
        "vectorstore": RETRIEVE,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

# Set memory
memory = MemorySaver()

# Compile Graph
graph = workflow.compile(checkpointer=memory)

# Save Graph DAG
#if not os.path.isfile("assests/graph.png"):
#    logger.info(f"Saving Graph DAG to {'assests/graph.png'}")
#    graph.get_graph().draw_mermaid_png(output_file_path="assets/graph.png")
