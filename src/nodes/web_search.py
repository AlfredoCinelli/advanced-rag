# Import packages and modules

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from src.state import GraphState
from src.chains.tool_agent import tool_agent_executor
from src.utils.logging import logger
import warnings

warnings.filterwarnings("ignore")
load_dotenv("local/.env")

web_search_tool = TavilySearchResults(
    max_results=3,
)

# Define Web Search Node
def web_search_node(
    state: GraphState,
) -> dict[str, list[Document]]:
    """
    Function defining the web search node.

    :param state: state of the Graph
    :type state: GraphState
    :return: dictionary containing the question and the documents retrieved
    :rtype: dict[str, str | list[Document] | Document]
    """
    logger.info("Performing web search...")
    question = state.get("question")
    documents = state.get("documents")

    tavily_results = web_search_tool.invoke({"query": question})
    logger.info(f"Fetched {len(tavily_results)} results from the web.")
    joined_tavily_results = "\n".join(
        [tavily_result.get("content", "") for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_results)

    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "question": question,
        "documents": documents,
    }

# Define Function Calling Web Search Node
def agent_search_node(
    state: GraphState,
) -> dict[str, list[Document]]:
    """
    Function defining the search node.
    The Node is made of a Tool calling Agent implemented via a Langchain chain.

    :param state: state of the Graph
    :type state: GraphState
    :return: dictionary containing the question and the documents retrieved
    :rtype: dict[str, str | list[Document] | Document]
    """
    logger.info("Performing web search...")
    question = state.get("question")
    documents = state.get("documents")

    logger.info("Calling tool agent...")
    tool_agent_result = tool_agent_executor.invoke(input={"query": question})
    web_results = Document(page_content=tool_agent_result.get("output", ""))

    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]

    return {
        "question": question,
        "documents": documents,
    }
