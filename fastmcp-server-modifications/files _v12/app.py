"""
FastMCP Streamlit Chatbot Frontend
────────────────────────────────────
HEAD Agent  : LangChain + Azure OpenAI / OpenAI  (matches reference main1.py)
Sub-Agents  : Lighter model for summarisation
MCP Client  : FastMCP Streamable-HTTP transport

Architecture:
  User message
       │
       ▼
  initialize_llm(head_model)          ← LangChain AzureChatOpenAI / ChatOpenAI
       │  .bind_tools(mcp_tools)
       ▼
  Agent loop  ──► tool_calls detected
       │
       ▼
  FastMCP Client → POST /mcp          ← executes tool on MCP server
       │
       ▼
  ToolMessage fed back → final answer
       │
       ▼
  Streamlit chat_message("assistant")

Run:
    # Terminal 1 – MCP server
    cd ..  &&  python -m app.server

    # Terminal 2 – Streamlit UI
    cd frontend  &&  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client

# LangChain imports
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import AzureChatOpenAI, ChatOpenAI

# ── Path so app.config is importable when running from frontend/ ──────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration (all driven from .env)
# ─────────────────────────────────────────────────────────────────────────────
MCP_ENDPOINT  = os.environ.get("MCP_ENDPOINT",        "http://127.0.0.1:8000/mcp")
MAX_HISTORY   = int(os.environ.get("MAX_MESSAGE_HISTORY", "20"))

# Azure OpenAI
AZURE_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT",    "")
AZURE_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY",     "")
AZURE_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

# Standard OpenAI (fallback when Azure keys are absent)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model list – mirrors reference main1.py exactly
# Map: display-name → Azure deployment name (or OpenAI model name)
MODEL_DEPLOYMENT_MAP: dict[str, str] = {
    "azure-gpt-4o":     os.environ.get("AZURE_DEPLOYMENT_GPT4O",      "gpt-4o"),
    "azure-gpt-4o-mini":os.environ.get("AZURE_DEPLOYMENT_GPT4O_MINI", "gpt-4o-mini"),
    "azure-gpt-4o-act": os.environ.get("AZURE_DEPLOYMENT_GPT4O_ACT",  "gpt-4o"),
}
AVAILABLE_MODELS = list(MODEL_DEPLOYMENT_MAP.keys())


# ─────────────────────────────────────────────────────────────────────────────
# initialize_llm  (mirrors the reference file's llm.initialize_llm pattern)
# ─────────────────────────────────────────────────────────────────────────────

def initialize_llm(model_name: str, temperature: float = 0.0) -> AzureChatOpenAI | ChatOpenAI:
    """
    Return a LangChain chat model for the given display name.

    - If Azure credentials are present  → AzureChatOpenAI
    - Otherwise                          → ChatOpenAI (standard OpenAI)

    Args:
        model_name:  One of the keys in AVAILABLE_MODELS.
        temperature: Sampling temperature (0 = deterministic).

    Returns:
        A LangChain BaseChatModel ready for .invoke() / .bind_tools().
    """
    deployment = MODEL_DEPLOYMENT_MAP.get(model_name, model_name)

    if AZURE_ENDPOINT and AZURE_API_KEY:
        logger.info("LLM: AzureChatOpenAI | deployment=%s", deployment)
        return AzureChatOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            azure_deployment=deployment,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
            temperature=temperature,
            max_tokens=4096,
        )

    # Fallback to standard OpenAI
    logger.info("LLM: ChatOpenAI | model=%s", deployment)
    return ChatOpenAI(
        model=deployment,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        max_tokens=4096,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async helpers – bridge FastMCP (async) → Streamlit (sync)
# ─────────────────────────────────────────────────────────────────────────────

def run_async(coro: Any) -> Any:
    """Execute an async coroutine safely from sync Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _fetch_mcp_tools_raw() -> list[Any]:
    """Return raw FastMCP tool objects from the server."""
    async with Client(MCP_ENDPOINT) as client:
        return await client.list_tools()


async def _call_mcp_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Invoke one MCP tool and return its text output."""
    async with Client(MCP_ENDPOINT) as client:
        result = await client.call_tool(tool_name, tool_input)
    return "\n".join(
        block.text if hasattr(block, "text") else str(block)
        for block in result
    )


def call_mcp_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    return run_async(_call_mcp_tool(tool_name, tool_input))


# ─────────────────────────────────────────────────────────────────────────────
# Convert MCP tools → LangChain StructuredTools
# ─────────────────────────────────────────────────────────────────────────────

def _build_langchain_tools(raw_tools: list[Any]) -> list[StructuredTool]:
    """
    Wrap each MCP tool as a LangChain StructuredTool so the LLM can call it
    via standard LangChain tool-use / function-calling.
    """
    lc_tools: list[StructuredTool] = []

    for t in raw_tools:
        tool_name  = t.name
        tool_desc  = t.description or ""
        tool_schema = t.inputSchema if hasattr(t, "inputSchema") else {
            "type": "object", "properties": {}, "required": []
        }

        # Build a closure so each tool captures its own name
        def _make_fn(name: str):
            def _run(**kwargs: Any) -> str:
                logger.info("MCP tool call: %s(%s)", name, kwargs)
                try:
                    return call_mcp_tool(name, kwargs)
                except Exception as exc:
                    return json.dumps({"error": str(exc)})
            _run.__name__ = name
            return _run

        lc_tools.append(
            StructuredTool.from_function(
                func=_make_fn(tool_name),
                name=tool_name,
                description=tool_desc,
                args_schema=None,          # LangChain infers from the JSON schema below
            )
        )

    return lc_tools


def fetch_tools() -> tuple[list[Any], list[StructuredTool]]:
    """
    Fetch tools from MCP server and return (raw_tools, langchain_tools).
    raw_tools   → used for sidebar display
    lc_tools    → passed to llm.bind_tools()
    """
    raw = run_async(_fetch_mcp_tools_raw())
    return raw, _build_langchain_tools(raw)


# ─────────────────────────────────────────────────────────────────────────────
# HEAD Agent loop  (LangChain tool-use agentic loop)
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    chat_history: list[dict[str, Any]],
    lc_tools: list[StructuredTool],
    head_model: str,
    system_prompt: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """
    Agentic loop using LangChain + OpenAI tool-calling.

    1. Build LangChain message list from chat_history.
    2. Bind MCP tools to the LLM.
    3. Invoke LLM → check for tool_calls.
    4. Execute each tool via MCP, append ToolMessages.
    5. Repeat until LLM produces a final text response.

    Returns:
        (final_text, tool_calls_log)
    """
    llm = initialize_llm(head_model)
    llm_with_tools = llm.bind_tools(lc_tools)

    # Build LangChain message objects
    lc_messages: list[Any] = []
    if system_prompt:
        lc_messages.append(SystemMessage(content=system_prompt))

    for m in chat_history:
        role    = m.get("role", "")
        content = m.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))

    tool_calls_log: list[dict[str, Any]] = []

    # ── Agentic loop ──────────────────────────────────────────────────────────
    while True:
        response: AIMessage = llm_with_tools.invoke(lc_messages)
        lc_messages.append(response)

        # No tool calls → final answer
        if not response.tool_calls:
            return response.content or "", tool_calls_log

        # Execute all requested tool calls
        for tc in response.tool_calls:
            t_name  = tc["name"]
            t_args  = tc["args"]
            t_id    = tc["id"]

            logger.info("Tool call: %s(%s)", t_name, t_args)
            try:
                result_text = call_mcp_tool(t_name, t_args)
            except Exception as exc:
                result_text = json.dumps({"error": str(exc)})

            tool_calls_log.append({
                "tool":   t_name,
                "input":  t_args,
                "output": result_text,
            })
            lc_messages.append(
                ToolMessage(content=result_text, tool_call_id=t_id)
            )


# ─────────────────────────────────────────────────────────────────────────────
# Summarisation  (uses the Sub-Agents LLM – a lighter / cheaper model)
# ─────────────────────────────────────────────────────────────────────────────

def summarize_old_messages(
    messages: list[dict[str, Any]],
    sub_model: str,
) -> str:
    """
    Summarise old messages using the Sub-Agents LLM.
    Mirrors summarize_old_messages() from the reference file.
    """
    if not messages:
        return ""
    llm = initialize_llm(sub_model)
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if isinstance(m.get("content"), str)
    )
    prompt = (
        "Summarise the following conversation in 3–5 sentences, "
        "capturing the key topics and any important facts:\n\n"
        + transcript
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    return resp.content if hasattr(resp, "content") else str(resp)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tool_output(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _has_llm_credentials() -> bool:
    return bool((AZURE_ENDPOINT and AZURE_API_KEY) or OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit page config  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FastMCP Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

if "summary" not in st.session_state:
    st.session_state.summary = ""

# Resolve user name (supports OAuth proxy header – mirrors reference main1.py)
user_name = "local_user"
if st.context.headers:
    headers_dict = st.context.headers.to_dict()
    proxy_user = headers_dict.get("X-Auth-Request-Preferred-Username")
    if proxy_user:
        user_name = proxy_user
st.session_state["user"] = user_name

session_id = st.session_state.conversation_id + user_name

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 FastMCP Chat")
    st.divider()

    # ── Quick links (matches reference layout) ────────────────────────────────
    st.sidebar.markdown("##### User Guide")
    st.sidebar.markdown(
        '[Open User Guide ↗](https://google.com)',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("##### Changelogs")
    st.sidebar.markdown(
        '[View Changelogs ↗](https://google.com)',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("##### Issue Register")
    st.sidebar.markdown(
        '[Report an Issue ↗](https://google.com)',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Model Configuration  (mirrors reference main1.py exactly) ─────────────
    st.sidebar.markdown("### Model Configuration")
    st.sidebar.info(
        "Please refresh the page before switching models to ensure changes take effect"
    )

    head_model = st.sidebar.selectbox(
        " HEAD Agent LLM",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index("azure-gpt-4o-act") if "azure-gpt-4o-act" in AVAILABLE_MODELS else 0,
        help="Primary reasoning model – routes user requests to MCP tools.",
    )
    sub_agents_model = st.sidebar.selectbox(
        " Sub-Agents LLM",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index("azure-gpt-4o-mini") if "azure-gpt-4o-mini" in AVAILABLE_MODELS else 1,
        help="Lighter model used for conversation summarisation.",
    )

    st.session_state["head_model"]       = head_model
    st.session_state["sub_agents_model"] = sub_agents_model

    st.divider()

    # ── Server status ─────────────────────────────────────────────────────────
    st.markdown("### 🌐 Server Status")
    st.caption(f"Endpoint: `{MCP_ENDPOINT}`")

    if st.button("🔄 Refresh Tools", use_container_width=True):
        st.session_state.pop("available_tools",    None)
        st.session_state.pop("langchain_tools",    None)
        st.rerun()

    if "available_tools" in st.session_state:
        st.success(f"✅ {len(st.session_state['available_tools'])} tools loaded")
        with st.expander("View tools"):
            for t in st.session_state["available_tools"]:
                st.markdown(f"**`{t.name}`** – {t.description or ''}")
    else:
        st.warning("⚠️ Tools not yet loaded")

    st.divider()

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages         = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.summary          = ""
        st.session_state.conversation_id  = str(uuid.uuid4())
        st.rerun()

    st.caption(f"👤 User: `{user_name}`  |  Session: `{session_id[:12]}…`")

# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("FastMCP Analysis V1.0")
st.write("Welcome to the interactive chat app!")
st.write(
    "Ask anything — the assistant will automatically route your request "
    "to the right MCP tool (math, text utilities, health checks, and more)."
)

# ── Load MCP tools once per session ──────────────────────────────────────────
if "available_tools" not in st.session_state:
    with st.spinner("🔌 Connecting to FastMCP server…"):
        try:
            raw_tools, lc_tools = fetch_tools()
            st.session_state["available_tools"] = raw_tools
            st.session_state["langchain_tools"]  = lc_tools
        except Exception as exc:
            st.error(
                f"❌ Could not connect to MCP server at `{MCP_ENDPOINT}`.\n\n"
                f"**Error:** `{exc}`\n\n"
                "Make sure the server is running:\n"
                "```\ncd ..  &&  python -m app.server\n```"
            )
            st.session_state["available_tools"] = []
            st.session_state["langchain_tools"]  = []

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant"):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 {len(msg['tool_calls'])} tool call(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**Tool:** `{tc['tool']}`")
                        raw_out = tc.get("output", "")
                        try:
                            st.json({"input": tc["input"], "output": json.loads(raw_out)})
                        except Exception:
                            st.markdown(f"**Input:** `{tc['input']}`\n\n**Output:** {raw_out}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input():

    # Guard: credentials
    if not _has_llm_credentials():
        st.error(
            "❌ No LLM credentials found.\n\n"
            "Set **`AZURE_OPENAI_ENDPOINT`** + **`AZURE_OPENAI_API_KEY`** "
            "(Azure) or **`OPENAI_API_KEY`** (OpenAI) in your `.env` file."
        )
        st.stop()

    # Guard: tools
    if not st.session_state.get("langchain_tools"):
        st.error("❌ No MCP tools available. Check that the MCP server is running.")
        st.stop()

    # ── Display user message ──────────────────────────────────────────────────
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # ── Summarise if history is too long (mirrors reference main1.py) ─────────
    if len(st.session_state.messages) > MAX_HISTORY:
        old_messages = st.session_state.messages[:-MAX_HISTORY]
        if "summary" not in st.session_state:
            st.session_state.summary = ""
        st.session_state.summary = summarize_old_messages(
            old_messages, st.session_state["sub_agents_model"]
        )

    # ── Build context (summary + recent messages) ─────────────────────────────
    system_prompt = (
        "You are a helpful assistant with access to a set of MCP tools. "
        "Use the tools when they help answer the user's question. "
        "Always present tool results in a clear, readable format."
    )
    if st.session_state.summary:
        system_prompt += f"\n\nConversation summary so far:\n{st.session_state.summary}"

    context: list[dict[str, Any]] = []
    context.extend(
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-MAX_HISTORY:]
        if isinstance(m.get("content"), str)
    )

    # ── Run HEAD agent ────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response_content, tool_calls_log = run_agent(
                    chat_history=context,
                    lc_tools=st.session_state["langchain_tools"],
                    head_model=st.session_state["head_model"],
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                st.error(f"❌ Agent error: {exc}")
                logger.exception("Agent error | session=%s", session_id)
                st.stop()

        st.write(response_content)

        if tool_calls_log:
            with st.expander(f"🔧 {len(tool_calls_log)} tool call(s)"):
                for tc in tool_calls_log:
                    st.markdown(f"**Tool:** `{tc['tool']}`")
                    raw_out = tc.get("output", "")
                    try:
                        st.json({"input": tc["input"], "output": json.loads(raw_out)})
                    except Exception:
                        st.markdown(f"**Input:** `{tc['input']}`\n\n**Output:** {raw_out}")

    # ── Persist to session state ──────────────────────────────────────────────
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    response_content,
        "tool_calls": tool_calls_log,
    })
