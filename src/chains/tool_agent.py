"""Module defining a Tool calling Agent Chain."""

# Import packages and modules

import os
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents import tool
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from tavily import TavilyClient
from langchain_core.tools import Tool
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
import warnings
from datetime import datetime
from src.utils.logging import logger
from src.constants import FUNCTION_CALLER_TEMPLATE

warnings.filterwarnings("ignore")
load_dotenv("local/.env")


# Define Tools
@tool(parse_docstring=False)
def wikipedia_search(
    query: str,
) -> str:
    """
    Use Wikipedia to search for a query.
    This is useful to search for knowledge about a topic not in real time.

    :param query: query to search for
    :type query: str
    :return: summary of retrieved Wikipedia pages
    :rtype: str
    """
    logger.info("Calling Wikipedia fetcher tool.")
    wikipedia_res = WikipediaLoader(
        query=query,
        load_max_docs=3,
        doc_content_chars_max=2_000,
    ).load()

    wikipedia_summary = "\n\n".join(
        [doc.metadata.get("summary", "") for doc in wikipedia_res]
    )

    return wikipedia_summary


@tool(parse_docstring=False)
def tavily_search(
    query: str,
) -> str:
    """
    Use Tavily to search for a query.
    This is useful to search for knowledge about a topic in real time.

    :param query: query to search for
    :type query: str
    :return: retrieved Tavily search results
    :rtype: str
    """
    logger.info("Calling Tavily search tool.")
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    return tavily_client.qna_search(query=query, max_results=5)


tools = [
    wikipedia_search,
    tavily_search,
]


def parse_tools(
    tools: list[Tool],
) -> list[str]:
    """
    Function to parse tool names from a list of tools.

    :param tools: list of Langchain tools
    :type tools: list[Tool]
    :return: list of Tool names
    :rtype: list[str]
    """
    return [tool.name for tool in tools]


# Assemble prompt
prompt = PromptTemplate(
    template=FUNCTION_CALLER_TEMPLATE,
    input_variables=["query"],
    partial_variables={
        "tools": parse_tools(tools),
        "today": datetime.today().strftime("%Y-%m-%d"),
    },
)

# Define LLM
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.0,
)

# Create Tool caller Agent
tool_agent: RunnableSequence = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

tool_agent_executor = AgentExecutor(
    agent=tool_agent,
    tools=tools,
    verbose=False,
)
