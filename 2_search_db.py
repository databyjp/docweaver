from docweaver.agents import docs_search_agent, DocSearchDeps
from docweaver.db import connect
import asyncio


async def main():
    query = "What is the replication architecture of Weaviate?"
    response = await docs_search_agent.run(
        f"Is there anything relevant to {query}?",
        deps=DocSearchDeps(client=connect())
    )
    for o in response.output:
        print(o)


if __name__ == "__main__":
    asyncio.run(main())
