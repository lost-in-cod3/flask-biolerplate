from dotenv import load_dotenv
import logging
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from graph.agent_state import AgentState, StreamingAgentState
from graph.builder import build_graph, build_secondary_graph
from utils.bm25_helper import custom_tokenizer
from langchain.callbacks import AsyncIteratorCallbackHandler
import pandas as pd
import httpx
import ssl
import os

import uuid

import streamlit as st

from typing import List, Dict

from datetime import datetime

from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Chroma

from chromadb import Client, Settings

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()


logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


load_dotenv(override=True)

client_settings = Settings(anonymized_telemetry=False)

# litellm

lite_llm_key = os.environ["LITE_LLM_KEY"]

lite_llm_url = os.environ["LITE_LLM_URL"]

lite_llm_embedding_model = os.environ["AZ_EMB_DEPLOYMENT"]

ctx = ssl.create_default_context()

ctx.check_hostname = False

ctx.verify_mode = ssl.CERT_REQUIRED

http_client = httpx.Client(verify=ctx)

rag_tool_embeddings = OpenAIEmbeddings(
    api_key=lite_llm_key,
    base_url=lite_llm_url,
    http_client=http_client,
    model=lite_llm_embedding_model,
)

#

# Sidebar

st.sidebar.markdown("""
The app is designed to help analyzing operational data
""")

st.sidebar.markdown("##### User Guide")
st.sidebar.markdown("##### Changelogs")


# #Streamlit page config
#

st.set_page_config(page_title="Askops Incident Analysis", layout="centered")
st.title("Askops Incident Analysis V2.01")

user_name = "Local_TEST"

if st.context.headers:
    headers_dict = st.context.headers.to_dict()
    st.session_state.user = headers_dict.get(
        "X-Auth-Request-Preferred-Username")
    if st.session_state.user is not None:
        user_name = st.session_state.get("user")

# Initialize memory and tools
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
session_id = st.session_state.conversation_id + user_name
st.session_state.session_id = session_id

# Create typed variables
corpus_docs: List[Document] = []
bm25_cache: Dict[str, BM25Retriever] = {}

# never empty but corpus_docs will be overwriten
bm25_cache["bm25::global"] = BM25Retriever.from_documents(
    [Document(page_content="No data found")], preprocess_func=custom_tokenizer)

# Initialize session_state if missing
if "corpus_docs" not in st.session_state:
    st.session_state["corpus_docs"] = corpus_docs

if "bm25_cache" not in st.session_state:
    st.session_state["bm25_cache"] = bm25_cache


# LLMs & helpers
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = Chroma(
        embedding_function=rag_tool_embeddings,
        client_settings=client_settings,
        collection_metadata={"hnsw:space": "cosine"},
    )

# LangGraph assembly

compiled_graph = build_graph(
    st.session_state.vectorstore, user_name, session_id)
streaming_graph = build_secondary_graph(
    st.session_state.vectorstore, st.session_state.bm25_cache, st.session_state.corpus_docs, user_name, session_id)

# Streamlit UI
# Initialize session_state for conversation history

if "messages" not in st.session_state:
    st.session_state.messages = [("role": "assistant", "content": "How can I help you?")]

# Display the conversation history

for msg in st.session_state.messages:
    if msg["role"] in ["user", "assistant"] and msg.get("content"):
        st.chat_message(msg["role"]).write(msg["content"])

if "graph" not in st.session_state:
    st.session_state.graph = compiled_graph

if "streaming_graph" not in st.session_state:
    st.session_state.streaming_graph = streaming_graph

# user_input

if user_input := st.chat_input():
    cb = AsyncIteratorCallbackHandler()

    # Add user message to chat history

    st.session_state.messages.append(("role": "user", "content": user_input))

    with st.chat_message("user"):
        st.write(user_input)

    init_state: AgentState = {
        "user": user_name,
        "session id": session id,
        "question": user_input,
        "messages": st.session_state.messages,
        "route": "general",
        "answer": None,
        "reasoning": None,
        "rag query": ""
        "rag filter":
    }
    result: AgentState = st.session_state.graph.invoke(init_state)
    logger.info(result["answer"])

    if result["answer"]:
        with st.chat_message("assistant"):
            answer_ph = st.empty()
            answer = result["answer"]
            answer_ph.markdown(answer)
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]})
    else:
        # need a different message Type
        logger.info("trigger streaming response")
        init_state: StreamingAgentState = result
        final_answer = ""
        with st.chat_message("assistant"):
            answer_ph = st.empty()
            for mode, chunk in st.session_state.streaming_graph.stream(
                # Stream both 'updates' and 'messages' modes
                init_state,
                stream_mode=["values", "messages"],
            ):

                if mode == "values":
                    if chunk.get("answer") is not None:
                        final_answer = chunk["answer"]
                        answer_ph.markdown(final_answer)
                elif mode == "messages":
                    token, meta=chunk
                    if token:
                        final_answer += token.content
                        answer_ph.write(final_answer)

        # Persist assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer})
