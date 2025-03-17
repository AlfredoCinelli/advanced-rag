"""Module building the frontend of the Advanced RAG via Streamlit."""

import warnings
from dotenv import load_dotenv

import streamlit as st
import random
import torch

from src.graph import graph

torch.classes.__path__ = []

load_dotenv("local/.env")
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(page_title="Advanced-RAG", layout="wide")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    st.session_state.graph = graph

if "config" not in st.session_state:
    st.session_state.config = {"configurable": {"thread_id": str(random.randint(0, 1000))}}

# Display chat title
st.title("🤖 Advanced-RAG")
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar= avatar):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get assistant response
    with st.spinner("🗣️ Calling RAG Workflow, ☕ it can take some time..."):
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response = ""

            try:
                result = st.session_state.graph.invoke(
                    {
                        "question": prompt, # user question
                        "iterations": 0, # reset generations iterations
                    },
                    st.session_state.config,
                )

                # Process response
                response = result.get("generation")
                message_placeholder.markdown(response + "▌")

                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

            except Exception as exc:
                st.error(f"An error occurred: {str(exc)}")


with st.sidebar:
    with st.expander("⚙️ Tools"):
        st.caption(
            "- Pinecone Vector Database: vector database used to store and retrieve KB via semantic search."
        )
        st.caption(
            "- Wikipedia Search: tool allowing to search for information on Wikipedia."
        )
        st.caption(
            "- Tavily Web Search: tool allowing to search for information on the web."
        )
    if st.sidebar.button("Clear Chat", help="Remove the chat history made so far!"):
        st.session_state.messages = []
        st.rerun()
