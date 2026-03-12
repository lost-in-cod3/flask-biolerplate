"""
FastMCP Streamlit Chatbot Frontend
────────────────────────────────────
Mirrors the GUI structure from the reference app (sidebar, model selector,
chat history, message summarisation) but replaces the agent graph with:

  User message
       │
       ▼
  Anthropic Claude  ──► decides which MCP tool(s) to call
       │
       ▼
  FastMCP Client (Streamable HTTP)  ──► executes tool on the server
       │
       ▼
  Final LLM response  ──► displayed in chat

Run:
    # Terminal 1 – start MCP server
    cd ..  &&  python -m app.server

    # Terminal 2 – start Streamlit
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

import anthropic
import streamlit as st
from dotenv import load_dotenv
from fastmcp import Client

# ── Path setup so we can import app.config when running from frontend/ ────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Load .env from project root ───────────────────────────────────────────────
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"), override=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MCP_ENDPOINT   = os.environ.get("MCP_ENDPOINT",       "http://127.0.0.1:8000/mcp")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_API_KEY",  "")
MAX_HISTORY    = int(os.environ.get("MAX_MESSAGE_HISTORY", "20"))

AVAILABLE_MODELS = [
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-4-5-20251001",
]

# ─────────────────────────────────────────────────────────────────────────────
# Async helpers (Streamlit is sync; we bridge with asyncio.run)
# ─────────────────────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from synchronous Streamlit code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


async def _fetch_tools() -> list[dict[str, Any]]:
    """Connect to MCP server and return tools as Anthropic tool definitions."""
    async with Client(MCP_ENDPOINT) as client:
        tools = await client.list_tools()
    return [
        {
            "name": t.name,
            "description": t.description or "",
            "input_schema": t.inputSchema if hasattr(t, "inputSchema") else {"type": "object", "properties": {}},
        }
        for t in tools
    ]


async def _call_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Call a single MCP tool and return its text result."""
    async with Client(MCP_ENDPOINT) as client:
        result = await client.call_tool(tool_name, tool_input)
    # FastMCP returns a list of content blocks
    parts = []
    for block in result:
        if hasattr(block, "text"):
            parts.append(block.text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


def fetch_tools() -> list[dict[str, Any]]:
    return run_async(_fetch_tools())


def call_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    return run_async(_call_tool(tool_name, tool_input))


# ─────────────────────────────────────────────────────────────────────────────
# LLM agent loop
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(
    user_messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    model: str,
    system_prompt: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """
    Run a single agent turn:
      1. Send messages + tools to Claude.
      2. If Claude requests tool calls, execute them via MCP.
      3. Feed results back to Claude for the final answer.

    Returns (final_text, tool_calls_log).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    tool_calls_log: list[dict[str, Any]] = []
    messages = list(user_messages)

    while True:
        kwargs: dict[str, Any] = dict(
            model=model,
            max_tokens=4096,
            messages=messages,
            tools=tools,
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = client.messages.create(**kwargs)

        # ── Collect text + tool_use blocks ───────────────────────────────────
        assistant_content: list[Any] = []
        text_parts: list[str] = []
        tool_use_blocks: list[Any] = []

        for block in response.content:
            assistant_content.append(block)
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        messages.append({"role": "assistant", "content": assistant_content})

        # ── No tool calls → final answer ─────────────────────────────────────
        if not tool_use_blocks or response.stop_reason == "end_turn":
            return " ".join(text_parts), tool_calls_log

        # ── Execute tool calls ────────────────────────────────────────────────
        tool_results: list[dict[str, Any]] = []
        for tb in tool_use_blocks:
            logger.info("Calling tool %s with %s", tb.name, tb.input)
            try:
                result_text = call_tool(tb.name, tb.input)
            except Exception as exc:
                result_text = json.dumps({"error": str(exc)})

            tool_calls_log.append({
                "tool": tb.name,
                "input": tb.input,
                "output": result_text,
            })
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_text,
            })

        messages.append({"role": "user", "content": tool_results})


# ─────────────────────────────────────────────────────────────────────────────
# Summarisation (keeps long conversations from blowing the context window)
# ─────────────────────────────────────────────────────────────────────────────

def summarise_messages(messages: list[dict[str, Any]], model: str) -> str:
    """Ask Claude to summarise a list of past messages into a short paragraph."""
    if not messages:
        return ""
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in messages
        if isinstance(m.get("content"), str)
    )
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": (
                "Summarise the following conversation in 3–5 sentences, "
                "capturing the key topics and any important facts:\n\n"
                + transcript
            ),
        }],
    )
    return resp.content[0].text if resp.content else ""


# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FastMCP Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 FastMCP Chat")
    st.divider()

    # ── Quick links ───────────────────────────────────────────────────────────
    st.markdown("##### 📖 User Guide")
    st.markdown("[Open User Guide ↗](https://docs.anthropic.com/en/docs/build-with-claude/mcp)", unsafe_allow_html=True)

    st.markdown("##### 📝 Changelogs")
    st.markdown("[View Changelogs ↗](https://github.com/jlowin/fastmcp/releases)", unsafe_allow_html=True)

    st.markdown("##### 🐛 Issue Register")
    st.markdown("[Report an Issue ↗](https://github.com/jlowin/fastmcp/issues)", unsafe_allow_html=True)

    st.divider()

    # ── Model configuration ───────────────────────────────────────────────────
    st.markdown("### ⚙️ Model Configuration")
    st.info("💡 Refresh the page before switching models to ensure changes take effect.")

    head_model = st.selectbox(
        "🧠 HEAD Agent LLM",
        AVAILABLE_MODELS,
        index=0,
        help="Model used for reasoning and routing user requests to tools.",
    )
    sub_agents_model = st.selectbox(
        "🔧 Sub-Agents LLM",
        AVAILABLE_MODELS,
        index=2,
        help="Lighter model used for summarisation and tool-result processing.",
    )

    st.session_state["head_model"]       = head_model
    st.session_state["sub_agents_model"] = sub_agents_model

    st.divider()

    # ── Server status ─────────────────────────────────────────────────────────
    st.markdown("### 🌐 Server Status")
    st.caption(f"Endpoint: `{MCP_ENDPOINT}`")

    if st.button("🔄 Refresh Tools", use_container_width=True):
        st.session_state.pop("available_tools", None)
        st.rerun()

    if "available_tools" in st.session_state:
        st.success(f"✅ {len(st.session_state['available_tools'])} tools loaded")
        with st.expander("View tools"):
            for t in st.session_state["available_tools"]:
                st.markdown(f"**`{t['name']}`** – {t['description']}")
    else:
        st.warning("⚠️ Tools not yet loaded")

    st.divider()

    # ── Clear conversation ────────────────────────────────────────────────────
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages  = [{"role": "assistant", "content": "How can I help you?"}]
        st.session_state.summary   = ""
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Main page
# ─────────────────────────────────────────────────────────────────────────────
st.title("FastMCP Analysis v1.0")
st.write("Welcome to the interactive FastMCP chat app!")
st.write(
    "Ask anything — the assistant will automatically route your request "
    "to the right MCP tool (math, text utilities, health checks, and more)."
)

# ── Session state init ────────────────────────────────────────────────────────
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

if "summary" not in st.session_state:
    st.session_state.summary = ""

# ── Load MCP tools once ───────────────────────────────────────────────────────
if "available_tools" not in st.session_state:
    with st.spinner("🔌 Connecting to FastMCP server…"):
        try:
            st.session_state.available_tools = fetch_tools()
        except Exception as exc:
            st.error(
                f"❌ Could not connect to MCP server at `{MCP_ENDPOINT}`.\n\n"
                f"**Error:** `{exc}`\n\n"
                "Make sure the server is running:\n"
                "```\ncd ..  &&  python -m app.server\n```"
            )
            st.session_state.available_tools = []

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] in ("user", "assistant"):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            # Show tool calls log if present
            if msg.get("tool_calls"):
                with st.expander(f"🔧 {len(msg['tool_calls'])} tool call(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**Tool:** `{tc['tool']}`")
                        st.json({"input": tc["input"], "output": _parse_tool_output(tc["output"])
                                 if (tc.get("output") or "").startswith("{") else tc["output"]})

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything…"):

    if not ANTHROPIC_KEY:
        st.error("❌ `ANTHROPIC_API_KEY` is not set. Add it to your `.env` file.")
        st.stop()

    if not st.session_state.available_tools:
        st.error("❌ No MCP tools available. Check that the server is running.")
        st.stop()

    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ── Summarise old messages if history is too long ─────────────────────────
    if len(st.session_state.messages) > MAX_HISTORY:
        old = st.session_state.messages[:-MAX_HISTORY]
        st.session_state.summary = summarise_messages(
            old, st.session_state["sub_agents_model"]
        )

    # ── Build context window ──────────────────────────────────────────────────
    system_prompt = (
        "You are a helpful assistant with access to a set of MCP tools. "
        "Use the tools when they help answer the user's question. "
        "Always present tool results in a clear, readable format."
    )
    if st.session_state.summary:
        system_prompt += f"\n\nConversation summary so far:\n{st.session_state.summary}"

    recent_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[-MAX_HISTORY:]
        if isinstance(m.get("content"), str)
    ]

    # ── Run the agent ─────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                response_text, tool_calls_log = run_agent(
                    user_messages=recent_messages,
                    tools=st.session_state.available_tools,
                    model=st.session_state["head_model"],
                    system_prompt=system_prompt,
                )
            except anthropic.AuthenticationError:
                st.error("❌ Invalid `ANTHROPIC_API_KEY`. Please check your `.env` file.")
                st.stop()
            except Exception as exc:
                st.error(f"❌ Agent error: {exc}")
                logger.exception("Agent error")
                st.stop()

        st.write(response_text)

        if tool_calls_log:
            with st.expander(f"🔧 {len(tool_calls_log)} tool call(s)"):
                for tc in tool_calls_log:
                    st.markdown(f"**Tool:** `{tc['tool']}`")
                    raw = tc.get("output", "")
                    try:
                        parsed = json.loads(raw)
                        st.json({"input": tc["input"], "output": parsed})
                    except Exception:
                        st.markdown(f"**Input:** `{tc['input']}`\n\n**Output:** {raw}")

    # Persist
    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "tool_calls": tool_calls_log,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Helper (used in history rendering above)
# ─────────────────────────────────────────────────────────────────────────────
def _parse_tool_output(raw: str) -> Any:
    try:
        return json.loads(raw)
    except Exception:
        return raw
