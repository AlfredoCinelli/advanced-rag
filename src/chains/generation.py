"""Module implementing the Generation chain."""

# Import packages and modules

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.utils.misc import format_docs
from src.constants import GENERATION_TEMPLATE

import warnings

warnings.filterwarnings("ignore")

# Define LLM
llm = ChatOllama(
    model="phi4",
    temperature=0.0,
    # callbacks=[LLMCallbackHandler()],
)

# Assemble prompt
generation_prompt = PromptTemplate(
    template=GENERATION_TEMPLATE,
    input_variables=["context", "question"],
)

# Assemble chain
generation_chain: RunnableSequence = (
    RunnableLambda(
        lambda x: {"context": format_docs(x["context"]), "question": x["question"]}
    )
    | generation_prompt
    | llm
    | StrOutputParser()
)
