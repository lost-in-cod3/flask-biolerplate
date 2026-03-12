"""
FastMCP server entry point.

Default transport: Streamable HTTP (MCP 2025-03-26 spec)
    python -m app.server
    # MCP endpoint: http://127.0.0.1:8000/mcp

SSE transport (legacy):
    MCP_TRANSPORT=sse python -m app.server

stdio transport (Claude Desktop):
    MCP_TRANSPORT=stdio python -m app.server
"""

from __future__ import annotations

import asyncio
import sys

from fastmcp import FastMCP

from app.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.tools import register_all_tools

# ── Bootstrap ────────────────────────────────────────────────────────────────
setup_logging()
logger = get_logger(__name__)
settings = get_settings()

# ── Create FastMCP instance ───────────────────────────────────────────────────
mcp = FastMCP(
    name=settings.mcp_server_name,
    instructions=(
        f"This is {settings.mcp_server_name} v{settings.mcp_server_version}. "
        "Use the available tools to interact with the server. "
        "Call `health_check` to verify connectivity, `server_info` for metadata, "
        "or use the math and text utility tools."
    ),
)

# ── Register all tool groups ──────────────────────────────────────────────────
register_all_tools(mcp)

# ── ASGI app (uvicorn / gunicorn entry point) ─────────────────────────────────
#   uvicorn app.server:http_app --host 127.0.0.1 --port 8000
http_app = mcp.http_app()


# ── CLI runner ────────────────────────────────────────────────────────────────
async def _run() -> None:
    transport = settings.mcp_transport
    host = settings.mcp_host
    port = settings.mcp_port

    logger.info(
        "Starting %s v%s | transport=%s env=%s",
        settings.mcp_server_name,
        settings.mcp_server_version,
        transport,
        settings.env,
    )

    if transport == "stdio":
        logger.info("Launching stdio transport (Claude Desktop / local clients)")
        await mcp.run_async(transport="stdio")

    elif transport == "sse":
        logger.info("SSE endpoint  : http://%s:%d/sse", host, port)
        logger.info("Inspector     : npx @modelcontextprotocol/inspector http://%s:%d/sse", host, port)
        await mcp.run_async(transport="sse", host=host, port=port)

    else:  # streamable-http (default)
        logger.info("MCP endpoint  : http://%s:%d/mcp  (Streamable HTTP)", host, port)
        logger.info("Streamlit UI  : cd frontend && streamlit run app.py")
        logger.info("Test client   : python test_client.py")
        logger.info("Inspector     : npx @modelcontextprotocol/inspector http://%s:%d/mcp", host, port)
        await mcp.run_async(transport="streamable-http", host=host, port=port)


if __name__ == "__main__":
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        logger.info("Server stopped by user.")
        sys.exit(0)
