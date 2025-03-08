"""
Module defining a Question Router chain.

More in detail the chain assess if the the question can be answered by
using the KB via similarity search or if it requires a web search.
"""

# Import packages and modules

from typing import Literal
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_ollama import ChatOllama
import warnings
from src.constants import (
    TOPICS,
    QUESTION_ROUTER_TEMPLATE,
)

warnings.filterwarnings("ignore")
load_dotenv("local/.env")

# Define LLM
llm = ChatOllama(
    model="qwen2.5",
    temperature=0.0,
)


# Define Route query
class RouteQuery(BaseModel):
    """Data source to use to answer the question."""

    data_source: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to vectorstore or websearch.",
    )


structured_llm_router = llm.with_structured_output(
    schema=RouteQuery,
    method="json_schema",
)

# Assemble prompt
router_prompt = PromptTemplate(
    template=QUESTION_ROUTER_TEMPLATE,
    input_variables=["question"],
    partial_variables={
        "topics": "- ".join([topic + "\n" for topic in TOPICS]),
    },
)

# Assemble chain
question_router: RunnableSequence = router_prompt | structured_llm_router
