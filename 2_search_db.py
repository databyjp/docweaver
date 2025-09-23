from docweaver.agents import docs_search_agent, DocSearchDeps
from docweaver.db import connect
from pathlib import Path
import asyncio
import json
from helpers import TECH_DESCRIPTION_RESHARDING


async def main():
    response = await docs_search_agent.run(
        f"Find documents that may need editing, for a this feature: {TECH_DESCRIPTION_RESHARDING}",
        deps=DocSearchDeps(client=connect()),
    )
    for o in response.output:
        print(o)

    logpath = Path("logs/doc_search_agent.log")
    logpath.parent.mkdir(parents=True, exist_ok=True)
    responses = [o.model_dump() for o in response.output]
    with logpath.open(mode="w") as f:
        json.dump(responses, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())
