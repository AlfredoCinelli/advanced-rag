# Import packages and modules

from itertools import chain
import os
import time

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import WebBaseLoader
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import warnings
from src.constants import (
    EMBEDDING_MODEL,
)
from src.utils.logging import logger

warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv("local/.env")

# Define URLs to scrape
URLS = [
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Initialize Pinecone client

logger.info("Initializing Pinecone...")
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    ssl_verify=False,
)

logger.info("Getting indexes...")
available_indexes_raw = pc.list_indexes().get("indexes")
available_indexes = [index.get("name") for index in available_indexes_raw]
index_name = os.getenv("INDEX_NAME")

if index_name not in available_indexes: # change execution condition
    logger.info("Index does not exist.")
    task = input("The index does not exist. Do you want to create it? (y/n): ")
    if task == "y":
        logger.info("Creating index...")
        dimension = input("Enter the dimension of the embeddings to store: ")
        metric = input("Enter the metric to use for the index (dotproduct or cosine): ")
        pc.create_index(
            name=index_name,
            dimension=int(dimension),
            metric=metric,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        time.sleep(10)
    else:
        logger.warning("Index is not existing, create it before ingesting documents.")
    task = input("Do you want to ingest documents? (y/n): ")
    if task == "y":
        # Get Docs from URLs
        logger.info("Getting docs from URLs...")
        docs: list[list[Document]] = [WebBaseLoader(url).load() for url in URLS] # return a list of Document objects (one per URL page)

        # Flatten the list of lists into a single list of Documents
        docs_flat: list[Document] = list(chain.from_iterable(docs)) # equivalent to [item for sublist in docs for item in sublist]
        # Set embedding model (using HuggingFace)
        logger.info("Setting up embedding model...")
        hf_embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
            show_progress=True,
            multi_process=False,
        )
        text_splitter = SemanticChunker(
            embeddings=hf_embedding_model,
            breakpoint_threshold_type="percentile",
            min_chunk_size=200,
        )
        logger.info("Splitting documents into chunks...")
        chunks: list[Document] = text_splitter.split_documents(docs_flat) # split the documents into chunks
        # Define Vector Store
        try:
            logger.info("Indexing documents...")
            index = pc.Index(index_name)
            vectorstore = PineconeVectorStore(
                index=index,
                embedding=hf_embedding_model,
            )
            vectorstore.add_documents(chunks)
        except Exception as exc:
            logger.error(f"Error indexing documents: {exc}")
else:
    logger.info("VDB already exists. Skipping indexing.")

logger.info("Done!")
