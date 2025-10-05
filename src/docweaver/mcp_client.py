"""MCP client for weaviate-docs-mcp server."""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class WeaviateDocsMCPClient:
    """Client for interacting with the weaviate-docs-mcp server."""

    def __init__(self, server_directory: str = None):
        """Initialize the MCP client.

        Args:
            server_directory: Path to weaviate-docs-mcp directory.
                             Defaults to ~/code/weaviate-docs-mcp
        """
        if server_directory is None:
            server_directory = Path.home() / "code" / "weaviate-docs-mcp"
        self.server_directory = Path(server_directory)

        # Load environment from weaviate-docs-mcp/.env
        env_file = self.server_directory / ".env"
        if env_file.exists():
            from dotenv import dotenv_values
            self.env = {**os.environ, **dotenv_values(env_file)}
        else:
            self.env = os.environ.copy()

    async def search_docs(
        self, query: str, return_type: str = "full_documents"
    ) -> list[dict[str, Any]]:
        """Search documentation using the MCP server.

        Args:
            query: Search query to find relevant documentation
            return_type: Format of results (default: "full_documents")

        Returns:
            List of search results with document content
        """
        logger.info(f"Searching docs via MCP with query: {query}")

        server_params = StdioServerParameters(
            command="uv",
            args=[
                "--directory",
                str(self.server_directory),
                "run",
                "weaviate-docs-mcp",
            ],
            env=self.env,
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # Call the search_docs tool
                result = await session.call_tool(
                    "search_docs", arguments={"query": query, "return_type": return_type}
                )

                # Parse the result - MCP returns TextContent
                if result.content and len(result.content) > 0:
                    text_content = result.content[0].text

                    # Check if it's an error message
                    if text_content.startswith("Error"):
                        logger.error(f"MCP server returned error: {text_content}")
                        raise RuntimeError(f"MCP server error: {text_content}")

                    try:
                        documents = json.loads(text_content)
                        logger.info(f"MCP search returned {len(documents)} documents")
                        return documents
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse MCP response as JSON: {text_content[:200]}")
                        raise RuntimeError(f"Invalid JSON response from MCP server: {e}")
                else:
                    logger.warning("MCP search returned no content")
                    return []
