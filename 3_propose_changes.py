from pathlib import Path
import json
from docweaver.agents import DocSearchReturn, doc_writer_agent
import asyncio
from helpers import TECH_DESCRIPTION_COLLECTION_ALIASES



async def main():
    logpath = Path("logs/search_agent.log")
    with logpath.open(mode="r") as f:
        data = json.load(f)

    doc_search_return = [DocSearchReturn(**item) for item in data]

    response = await doc_writer_agent.run(
        f"Propose doc changes for {TECH_DESCRIPTION_COLLECTION_ALIASES}"
    )


if __name__ == "__main__":
    asyncio.run(main())
