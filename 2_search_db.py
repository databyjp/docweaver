from docweaver.agents import docs_search_agent, DocSearchDeps
from docweaver.db import connect
from pathlib import Path
import asyncio


async def main():
    query = "What is the replication architecture of Weaviate?"
    response = await docs_search_agent.run(
        f"Is there anything relevant to {query}?",
        deps=DocSearchDeps(client=connect())
    )

    logpath = Path("logs/search_agent.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    for o in response.output:
        print(o)
        with logpath.open(mode="a") as f:
            f.write(str(o))


if __name__ == "__main__":
    asyncio.run(main())
