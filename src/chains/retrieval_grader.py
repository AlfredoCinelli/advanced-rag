"""
Module implementing the Retrieval Grader chain.

More in detail the chain assess if the retrieved documents are relevant to the question.
"""

# Import packages and modules

from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence
from src.constants import RETRIEVAL_GRADER_TEMPLATE

# Define LLM
llm = ChatOllama(
    model="mistral-nemo", # llama3.2:3b
    temperature=0.0,
)


# Define Grader Structure
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'.",
    )


structured_llm_doc_grader = llm.with_structured_output(
    schema=GradeDocuments,
    method="json_schema",
)

# Assemble prompt
grader_prompt = PromptTemplate(
    template=RETRIEVAL_GRADER_TEMPLATE,
    input_variables=["document", "question"],
)

# Assemble chain
retrieval_grader: RunnableSequence = grader_prompt | structured_llm_doc_grader
