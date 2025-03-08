"""Module containing miscellaneous utility functions."""

# Import packages and modules

from langchain.schema import Document

# Define functins
def format_docs(
    documents: list[Document],
) -> str:
    """
    Function to format the retrieved documents.

    :param documents: list of retrieved documents
    :type documents: list[Document]
    :return: _description_
    :rtype: str
    """
    formatted_docs = "\n\n".join([document.page_content for document in documents])

    return formatted_docs
