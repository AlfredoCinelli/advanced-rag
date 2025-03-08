# Import packages and modules
from langchain_core.documents import Document

# Define scenarios
def scenario(function_name: str) -> dict[str, any]:
    if function_name == "generation_node":
        return [
            {
                "chain_name": "generation_node",
                "state_name": "generation_node",
                "expected_output": {
                    "question": "User input question.",
                    "documents": [Document(page_content="This is context retrieved from the vector store or via web search.")],
                    "generation": "This is an LLM generated answer.",
                    "iterations": 1,
                },
            },
        ]
    elif function_name == "grader_node":
        return [
            # Yes scenario
            {
                "chain_name": "grader_node_yes",
                "state_name": "grader_node",
                "expected_outout": {
                    "question": "User input question.",
                    "documents": [Document(metadata={}, page_content="This is the content of a document.")],
                    "confidence_score": 1.0,
                },
            },
            # No scenario
            {
                "chain_name": "grader_node_no",
                "state_name": "grader_node",
                "expected_outout": {
                    "question": "User input question.",
                    "documents": [],
                    "confidence_score": 0.0,
                },
            },
        ]
    elif function_name == "retriever_node":
        return [
            {
                "chain_name": "retriever_node",
                "state_name": "retriever_node",
                "expected_output": {
                    "question": "User input question.",
                    "documents": [Document(metadata={}, page_content="This is the content of a document.")],
                },
            },
        ]
    elif function_name == "agent_search_node":
        return [
            # Scenario with documents already
            {
                "chain_name": "agent_search_node",
                "state_name": "agent_search_node_documents",
                "expected_output": {
                    "question": "User input question.",
                    "documents": [
                        Document(metadata={}, page_content="This is the content of a document."),
                        Document(metadata={}, page_content="This is the output of the agent search."),
                    ],
                },
            },
            # Scenario with no documents
            {
                "chain_name": "agent_search_node",
                "state_name": "agent_search_node_no_documents",
                "expected_output": {
                    "question": "User input question.",
                    "documents": [
                        Document(metadata={}, page_content="This is the output of the agent search."),
                    ],
                },
            },
        ]