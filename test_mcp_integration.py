#!/usr/bin/env python3
"""Test MCP integration for docweaver."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from docweaver.mcp_client import WeaviateDocsMCPClient


async def main():
    """Test the MCP client integration."""
    print("Testing MCP client integration...")
    print("=" * 60)

    # Create MCP client
    client = WeaviateDocsMCPClient()

    # Test search
    query = "How do I configure backups in Weaviate?"
    print(f"\nQuery: {query}")
    print("-" * 60)

    try:
        results = await client.search_docs(query)
        print(f"\nFound {len(results)} documents:")
        print("-" * 60)

        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc.get('path', 'Unknown path')}")
            print(f"   Title: {doc.get('title', 'No title')}")
            print(f"   Summary: {doc.get('summary', 'No summary')[:100]}...")
            print(f"   Topics: {', '.join(doc.get('topics', []))}")
            print(f"   Doctype: {doc.get('doctype', 'Unknown')}")
            print(f"   Content length: {len(doc.get('content', ''))} chars")
            print(f"   Referenced files: {len(doc.get('referenced_files', {}))}")

        print("\n" + "=" * 60)
        print("✅ MCP integration test successful!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
