# Import packages and modules

from unittest.mock import MagicMock
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain.agents import tool

# Define scenarios
def scenario(function_name: str) -> list[dict[str, any]]:
    if function_name == "retrieval_grader":
        return [
            # Relevant document
            {
                "question": "agent memory",
                "document": Document(
                    page_content="Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions."
                ),
                "expected_output": "yes",
            },
            # Irrelevant document
            {
                "question": "agent memory",
                "document": Document(
                    page_content="Linear algebra is a branch of mathematics that studies vectors, matrices, and linear transformations. It is used in many areas of mathematics, including geometry, analysis, and probability. Linear algebra is also used in many applications, such as computer graphics, physics, and engineering."
                ),
                "expected_output": "no",
            },
        ]
    elif function_name == "generation_chain":
        return [
            {
                "question": "agent memory",
                "documents": [
                    Document(
                        page_content="Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions."
                    )
                ],
                "mocked_llm": MagicMock(
                    model_name="model",
                    temperature=0.0,
                    invoke=MagicMock(return_value="Ciao"),
                ),
            }
        ]
    elif function_name == "parse_tools":
        tools = [
            Tool(
                name="tool_1",
                func=lambda x: x,
                description="tool_1_description",
            ),
            Tool(
                name="tool_2",
                func=lambda x: x,
                description="tool_2_description",
            ),
        ]
        return [
            {
                "tools": tools,
                "expected_output": ["tool_1", "tool_2"],
            }
        ]
    elif function_name == "tool_agent":
        # tavily_search = Tool(
        #    name="tavily_search",
        #    description="Search using Tavily for real time information.",
        #    func=lambda query: "LVMH stock price today is 700 euros.",
        # )
        # wikipedia_search = Tool(
        #    name="wikipedia_search",
        #    description="Search using Wikpedia for non real time information.",
        #    func=lambda query: "Enzo Ferrari is an italian car manufacturer.",
        # )
        # Define Tools
        # @tool(parse_docstring=False)
        # def wikipedia_search_mock(
        #    query: str,
        # ) -> str:
        #    """
        #    Use Wikipedia to search for a query.
        #    This is useful to search for knowledge about a topic not in real time.
        #
        #    :param query: query to search for
        #    :type query: str
        #    :return: summary of retrieved Wikipedia pages
        #    :rtype: str
        #    """
        #
        #    return "Enzo Ferrari is an italian car manufacturer."
        #
        #
        # @tool(parse_docstring=False)
        # def tavily_search_mock(
        #    query: str,
        # ) -> str:
        #    """
        #    Use Tavily to search for a query.
        #    This is useful to search for knowledge about a topic in real time.
        #
        #    :param query: query to search for
        #    :type query: str
        #    :return: retrieved Tavily search results
        #    :rtype: str
        #    """
        #
        #    return "LVMH stock price today is 100 euros."
        return [
            # Real time search
            {
                "question": "LVMH stock price today",
                # "tavily_search_mock": tavily_search_mock,
                # "wikipedia_search_mock": wikipedia_search_mock,
                "expected_output": "LVMH stock price today is 100 euros.",
            },
        ]
    elif function_name == "hallucination_grader":
        return [
            # Answer is grounded in the retrieved documents
            {
                "question": "agent memory",
                "documents": [
                    Document(
                        page_content="Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions."
                    ),
                    Document(
                        page_content="Memory allows agents to store and persist information to have a more human-like inner working."
                    ),
                ],
                "generation": "Agent memory is a retention, so a kind of memory, that allows an agent to store and persist information to have a more human-like inner working.",
                "expected_output": "yes",
            },
            # Answer is not grounded in the retrieved documents
            {
                "question": "agent memory",
                "documents": [
                    Document(
                        page_content="Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions."
                    ),
                    Document(
                        page_content="Memory allows agents to store and persist information to have a more human-like inner working."
                    ),
                ],
                "generation": "Cosine similarity is a measure of similarity between two vectors, it's a cornerstone in linera algebra that is taking more attention in new GenAI appications.",
                "expected_output": "no",
            },
        ]
    elif function_name == "answer_grader":
        return [
            # Answer is factual to the question
            {
                "question": "What is agent memory?",
                "generation": "Agent memory is a type of memory that allows an agent to store information about its environment and use that information to make decisions. It is a type of memory that is used by artificial intelligence systems to store information about the environment and use that information to make decisions.",
                "expected_output": "yes",
            },
            # Answer is not factual to the question
            {
                "question": "What is agent memory?",
                "generation": "Cosine similarity is a measure of similarity between two vectors, it's a cornerstone in linera algebra that is taking more attention in new GenAI appications.",
                "expected_output": "no",
            },
        ]
    elif function_name == "question_router":
        return [
            # Question related to indexed KB
            {
                "question": "What is few-shot prompting?",
                "expected_output": "vectorstore",
            },
            # Question not related to indexed KB (hence resort to web search)
            {
                "question": "Which is the EBITDA of Microsoft made over the 2024?",
                "expected_output": "websearch",
            },
        ]
