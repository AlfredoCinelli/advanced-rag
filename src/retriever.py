"""Module containing the Retriever used to perform similarity search."""

# Import packages and modules

import os

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from dotenv import load_dotenv
import warnings
from src.constants import (
    EMBEDDING_MODEL,
    SIMILARITY,
    RERANKER_MODEL,
    TOP_K,
)

warnings.filterwarnings("ignore")
load_dotenv("local/.env")

# Setup retriever
hf_embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
    show_progress=True,
    multi_process=False,
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    ssl_verify=False,
)

index = pc.Index(os.getenv("INDEX_NAME"))

base_retriever = PineconeVectorStore(
    index=index,
    embedding=hf_embedding_model,  # embedding model to be used
).as_retriever(
    **SIMILARITY,  # SIMILARITY (plain vector search) or MMR (vector search with MMR post processor)
)

# Setup Document Compressor for reranking
model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=model, top_n=TOP_K)
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)
